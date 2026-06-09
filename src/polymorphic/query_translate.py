"""
PolymorphicQuerySet support functions
"""

import copy
from functools import reduce
from operator import or_
from typing import Any

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import models
from django.db.models import Q, Subquery
from django.db.models.fields.related import ForeignObjectRel, RelatedField
from django.db.utils import DEFAULT_DB_ALIAS

from .utils import _lazy_ctype, _map_queryname_to_class, concrete_descendants

# These functions implement the additional filter- and Q-object functionality.
# They form a kind of small framework for easily adding more
# functionality to filters and Q objects.
# Probably a more general queryset enhancement class could be made out of them.

###################################################################################
# PolymorphicQuerySet support functions


def translate_polymorphic_filter_definitions_in_kwargs(
    queryset_model: type[models.Model], kwargs: dict[str, Any], using: str = DEFAULT_DB_ALIAS
) -> list[Q]:
    """
    Translate the keyword argument list for PolymorphicQuerySet.filter()

    Any kwargs with special polymorphic functionality are replaced in the kwargs
    dict with their vanilla django equivalents.

    For some kwargs a direct replacement is not possible, as a Q object is needed
    instead to implement the required functionality. In these cases the kwarg is
    deleted from the kwargs dict and a Q object is added to the return list.

    Modifies: kwargs dict
    Returns: a list of non-keyword-arguments (Q objects) to be added to the filter() query.
    """
    additional_args = []
    for field_path, val in kwargs.copy().items():  # `copy` so we're not mutating the dict
        new_expr = _translate_polymorphic_filter_definition(
            queryset_model, field_path, val, using=using
        )

        if isinstance(new_expr, tuple):
            # replace kwargs element
            del kwargs[field_path]
            kwargs[new_expr[0]] = new_expr[1]

        elif isinstance(new_expr, models.Q):
            del kwargs[field_path]
            additional_args.append(new_expr)

    return additional_args


def translate_polymorphic_Q_object(
    queryset_model: type[models.Model], potential_q_object: Q, using: str = DEFAULT_DB_ALIAS
) -> Q:
    def tree_node_correct_field_specs(my_model: type[models.Model], node: Q) -> Q:
        "process all children of this Q node"
        result = None
        for child in node.children:
            if isinstance(child, (tuple, list)):
                # this Q object child is a tuple => a kwarg like Q( instance_of=ModelB )
                key, val = child
                new_expr = _translate_polymorphic_filter_definition(
                    my_model, key, val, using=using
                )
                if new_expr is None:
                    new_expr = node.__class__((key, val))
                elif isinstance(new_expr, tuple):
                    new_expr = node.__class__((new_expr[0], new_expr[1]))
            elif isinstance(child, models.Q):
                # this Q object child is another Q object, recursively process
                new_expr = tree_node_correct_field_specs(my_model, child)
            else:
                new_expr = node.__class__(child)

            if result is None:
                result = new_expr
            else:
                if node.connector == node.__class__.OR:
                    result |= new_expr
                elif getattr(node.__class__, "XOR", None) and node.connector == getattr(node.__class__, "XOR"):
                    result ^= new_expr
                else:
                    result &= new_expr

        if result is None:
            result = node.__class__()

        if node.negated:
            result = ~result

        return result

    if isinstance(potential_q_object, models.Q):
        return tree_node_correct_field_specs(queryset_model, potential_q_object)

    return potential_q_object  # type: ignore[unreachable]


def translate_polymorphic_filter_definitions_in_args(
    queryset_model: type[models.Model], args: tuple[Q, ...], using: str = DEFAULT_DB_ALIAS
) -> list[Q]:
    """
    Translate the non-keyword argument list for PolymorphicQuerySet.filter()

    In the args list, we return all kwargs to Q-objects that contain special
    polymorphic functionality with their vanilla django equivalents.
    We traverse the Q object tree for this (which is simple).


    Returns: modified Q objects
    """

    return [translate_polymorphic_Q_object(queryset_model, q, using=using) for q in args]


def _translate_polymorphic_filter_definition(
    queryset_model: type[models.Model],
    field_path: str,
    field_val: Any,
    using: str = DEFAULT_DB_ALIAS,
) -> tuple[str, Any] | Q | None:
    """
    Translate a keyword argument (field_path=field_val), as used for
    PolymorphicQuerySet.filter()-like functions (and Q objects).

    A kwarg with special polymorphic functionality is translated into
    its vanilla django equivalent, which is returned, either as tuple
    (field_path, field_val) or as Q object.

    Returns: kwarg tuple or Q object or None (if no change is required)
    """

    # handle instance_of expressions or alternatively,
    # if this is a normal Django filter expression, return None
    if field_path == "instance_of":
        return create_instanceof_q(field_val, using=using)
    elif field_path == "not_instance_of":
        return create_instanceof_q(field_val, not_instance_of=True, using=using)
    elif "___" not in field_path:
        return None  # no change

    # filter expression contains '___' (i.e. filter for polymorphic field)
    # => get the model class specified in the filter expression
    newpath = field_path
    while "___" in newpath:
        translated = translate_polymorphic_field_path(queryset_model, newpath)
        if translated == newpath:
            break
        newpath = translated
    return (newpath, field_val)


def translate_polymorphic_field_path(queryset_model: type[models.Model], field_path: str) -> str:
    """
    Translate a field path from a keyword argument, as used for
    PolymorphicQuerySet.filter()-like functions (and Q objects).
    Supports leading '-' (for order_by args).

    E.g.: if queryset_model is ModelA, then "ModelC___field3" is translated
    into modela__modelb__modelc__field3.
    Returns: translated path (unchanged, if no translation needed)
    """
    classname, sep, pure_field_path = field_path.partition("___")
    if not sep or not classname:
        return field_path

    negated = False
    if classname[0] == "-":
        negated = True
        classname = classname.lstrip("-")

    relation_prefix = ""
    current_model = queryset_model

    if "__" in classname:
        # the user has app label prepended to class name via __ => use Django's get_model function
        # OR the user has a relation path, e.g. relation__ModelC
        appname, sep, cname = classname.partition("__")
        model = None
        try:
            model_candidate = apps.get_model(appname, cname)
            if issubclass(model_candidate, queryset_model):
                model = model_candidate
        except LookupError:
            pass

        if model is None:
            # Maybe it's a relation path!
            relation_path, sep, cname = classname.rpartition("__")
            try:
                for part in relation_path.split("__"):
                    field = current_model._meta.get_field(part)
                    current_model = field.related_model
                model = _map_queryname_to_class(current_model, cname)
                relation_prefix = relation_path + "__"
            except (FieldDoesNotExist, AttributeError) as e:
                raise FieldError(f"Model {appname}.{cname} does not exist and '{classname}' is not a valid relation path") from e

    else:
        # the user has only given us the class name via ___
        # => select the model from the sub models of the queryset base model

        # Test whether it's actually a regular relation__ _fieldname (the field starting with an _)
        # so no tripple ClassName___field was intended.
        try:
            # This also retreives M2M relations now (including reverse foreign key relations)
            field = queryset_model._meta.get_field(classname)

            if isinstance(field, (RelatedField, ForeignObjectRel)):
                # Can also test whether the field exists in the related object to avoid ambiguity between
                # class names and field names, but that never happens when your class names are in CamelCase.
                return field_path  # No exception raised, field does exist.
        except FieldDoesNotExist:
            pass

        model = _map_queryname_to_class(queryset_model, classname)

    basepath = _create_base_path(current_model, model)

    if negated:
        newpath = "-"
    else:
        newpath = ""

    if relation_prefix:
        newpath += relation_prefix

    newpath += basepath
    if basepath:
        newpath += "__"

    newpath += pure_field_path
    return newpath


def _create_base_path(baseclass: type[models.Model], myclass: type[models.Model]) -> str:
    # create new field path for expressions, e.g. for baseclass=ModelA, myclass=ModelC
    # 'modelb__modelc" is returned
    for b in myclass.__bases__:
        if b == baseclass:
            return _get_query_related_name(myclass)

        path = _create_base_path(baseclass, b)
        if path:
            if b._meta.abstract or b._meta.proxy:  # type: ignore[attr-defined]
                return _get_query_related_name(myclass)
            else:
                return f"{path}__{_get_query_related_name(myclass)}"
    return ""


def _get_query_related_name(myclass: type[models.Model]) -> str:
    for f in myclass._meta.local_fields:
        if isinstance(f, models.OneToOneField) and f.remote_field.parent_link:
            return f.related_query_name()

    # Fallback to undetected name,
    # this happens on proxy models (e.g. SubclassSelectorProxyModel)
    return myclass.__name__.lower()


def create_instanceof_q(
    modellist: type[models.Model] | list[type[models.Model]] | tuple[type[models.Model], ...],
    not_instance_of: bool = False,
    using: str = DEFAULT_DB_ALIAS,
) -> Q | None:
    """
    Helper function for instance_of / not_instance_of
    Creates and returns a Q object that filters for the models in modellist,
    including all subclasses of these models (as we want to do the same
    as pythons isinstance() ).
    .
    We recursively collect all __subclasses__(), create a Q filter for each,
    and or-combine these Q objects. This could be done much more
    efficiently however (regarding the resulting sql), should an optimization
    be needed.
    """
    if not modellist:
        return None

    if not isinstance(modellist, (list, tuple)):
        from .models import PolymorphicModel

        if issubclass(modellist, PolymorphicModel):
            modellist = [modellist]
        else:
            raise TypeError(
                "PolymorphicModel: instance_of expects a list of (polymorphic) "
                "models or a single (polymorphic) model"
            )

    lazy_cts, ct_ids = _get_mro_content_type_ids(modellist, using)
    q = Q()
    if lazy_cts:
        q |= Q(
            polymorphic_ctype__in=Subquery(
                # no need to pass using here
                ContentType.objects.filter(reduce(or_, lazy_cts)).values("pk")
            )
        )
    if ct_ids:
        q |= Q(polymorphic_ctype__in=ct_ids)
    if not_instance_of:
        q = ~q
    return q


def _get_mro_content_type_ids(
    models: list[type[models.Model]] | tuple[type[models.Model], ...], using: str
) -> tuple[list[Q], list[int]]:
    lazy: list[Q] = []
    ids: list[int] = []
    for model in models:
        cid = _lazy_ctype(model, using=using)
        ids.append(cid.pk) if isinstance(cid, ContentType) else lazy.append(cid)
        for descendent in concrete_descendants(model, include_proxy=True):
            cid = _lazy_ctype(descendent, using=using)
            ids.append(cid.pk) if isinstance(cid, ContentType) else lazy.append(cid)
    return lazy, ids
