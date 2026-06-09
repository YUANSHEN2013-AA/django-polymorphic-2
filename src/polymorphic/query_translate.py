"""
PolymorphicQuerySet support functions
"""

import copy
from collections import defaultdict
from functools import lru_cache, reduce
from operator import or_
from typing import Any

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import models
from django.db.models import Q, Subquery
from django.db.models.fields.related import ForeignObjectRel, RelatedField
from django.db.utils import DEFAULT_DB_ALIAS

from .utils import _lazy_ctype, concrete_descendants


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
    for field_path, val in kwargs.copy().items():
        new_expr = _translate_polymorphic_filter_definition(
            queryset_model, field_path, val, using=using
        )

        if isinstance(new_expr, tuple):
            del kwargs[field_path]
            kwargs[new_expr[0]] = new_expr[1]
        elif isinstance(new_expr, models.Q):
            del kwargs[field_path]
            additional_args.append(new_expr)

    return additional_args


def translate_polymorphic_Q_object(
    queryset_model: type[models.Model], potential_q_object: Q, using: str = DEFAULT_DB_ALIAS
) -> Q:
    if isinstance(potential_q_object, models.Q):
        return _translate_polymorphic_q_node(queryset_model, potential_q_object, using=using)

    return potential_q_object  # type: ignore[unreachable]


def _translate_polymorphic_q_node(
    queryset_model: type[models.Model], node: Q, using: str = DEFAULT_DB_ALIAS
) -> Q:
    translated_node = copy.copy(node)
    translated_node.children = [
        _translate_polymorphic_q_child(queryset_model, child, using=using)
        for child in node.children
    ]
    return translated_node


def _translate_polymorphic_q_child(
    queryset_model: type[models.Model], child: Any, using: str = DEFAULT_DB_ALIAS
) -> Any:
    if isinstance(child, models.Q):
        return _translate_polymorphic_q_node(queryset_model, child, using=using)

    if isinstance(child, (tuple, list)) and len(child) == 2 and isinstance(child[0], str):
        key, val = child
        translated_child = _translate_polymorphic_filter_definition(
            queryset_model, key, val, using=using
        )
        if translated_child is None:
            return child
        return translated_child

    return child


def translate_polymorphic_filter_definitions_in_args(
    queryset_model: type[models.Model], args: tuple[Q, ...], using: str = DEFAULT_DB_ALIAS
) -> list[Q]:
    """
    Translate the non-keyword argument list for PolymorphicQuerySet.filter()

    In the args list, we return all kwargs to Q-objects that contain special
    polymorphic functionality with their vanilla django equivalents.
    We traverse the Q object tree for this.

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
    if field_path == "instance_of":
        return create_instanceof_q(field_val, using=using)
    if field_path == "not_instance_of":
        return create_instanceof_q(field_val, not_instance_of=True, using=using)
    if "___" not in field_path:
        return None

    return (translate_polymorphic_field_path(queryset_model, field_path), field_val)


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

    if "__" in classname:
        appname, _, model_name = classname.partition("__")
        model = _get_query_lookup_model_from_app_label(queryset_model, appname, model_name)
    else:
        if _is_relationship_field(queryset_model, classname):
            return field_path
        model = _get_query_lookup_model(queryset_model, classname)

    basepath = _create_base_path(queryset_model, model)
    newpath = "-" if negated else ""
    if basepath:
        newpath += f"{basepath}__{pure_field_path}"
    else:
        newpath += pure_field_path
    return newpath


@lru_cache(maxsize=None)
def _get_query_lookup_models(queryset_model: type[models.Model]) -> tuple[type[models.Model], ...]:
    result: list[type[models.Model]] = []
    seen: set[type[models.Model]] = set()

    for model in queryset_model.mro():
        if not isinstance(model, type) or not issubclass(model, models.Model):
            continue
        if model is models.Model:
            continue
        if model not in seen:
            seen.add(model)
            result.append(model)

    for model in concrete_descendants(queryset_model, include_proxy=True):
        if model not in seen:
            seen.add(model)
            result.append(model)

    return tuple(result)


def _get_query_lookup_model(
    queryset_model: type[models.Model], model_name: str
) -> type[models.Model]:
    matches: dict[str, list[type[models.Model]]] = defaultdict(list)
    for model in _get_query_lookup_models(queryset_model):
        matches[model.__name__.lower()].append(model)

    matched_models = matches.get(model_name.lower(), [])
    if len(matched_models) == 1:
        return matched_models[0]
    if len(matched_models) > 1:
        raise FieldError(
            f"{model_name} could refer to any of {[m._meta.label for m in matched_models]}. In "
            f"this case, please use the syntax: applabel__ModelName___field"
        )

    raise AssertionError(f"{model_name} is not a subclass of {queryset_model._meta.label}")


def _get_query_lookup_model_from_app_label(
    queryset_model: type[models.Model], appname: str, model_name: str
) -> type[models.Model]:
    try:
        model = apps.get_model(appname, model_name)
    except LookupError as exc:
        raise FieldError(f"Model {appname}.{model_name} does not exist") from exc

    if issubclass(model, queryset_model) or issubclass(queryset_model, model):
        return model

    raise FieldError(f"{model._meta.label} is not derived from {queryset_model._meta.label}")


def _is_relationship_field(queryset_model: type[models.Model], field_name: str) -> bool:
    try:
        field = queryset_model._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False

    return isinstance(field, (RelatedField, ForeignObjectRel))


def _create_base_path(baseclass: type[models.Model], myclass: type[models.Model]) -> str:
    if baseclass is myclass or issubclass(baseclass, myclass):
        return ""
    if issubclass(myclass, baseclass):
        return _create_descendant_base_path(baseclass, myclass)
    return ""


def _create_descendant_base_path(baseclass: type[models.Model], myclass: type[models.Model]) -> str:
    for base in myclass.__bases__:
        if not isinstance(base, type) or not issubclass(base, models.Model):
            continue

        if base == baseclass:
            return _get_query_related_name(myclass)

        path = _create_descendant_base_path(baseclass, base)
        if path:
            if base._meta.abstract or base._meta.proxy:  # type: ignore[attr-defined]
                return _get_query_related_name(myclass)
            return f"{path}__{_get_query_related_name(myclass)}"
    return ""


def _get_query_related_name(myclass: type[models.Model]) -> str:
    for f in myclass._meta.local_fields:
        if isinstance(f, models.OneToOneField) and f.remote_field.parent_link:
            return f.related_query_name()

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
