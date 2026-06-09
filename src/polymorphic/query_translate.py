"""
PolymorphicQuerySet support functions
"""

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


def translate_polymorphic_filter_definitions_in_kwargs(
    queryset_model: type[models.Model], kwargs: dict[str, Any], using: str = DEFAULT_DB_ALIAS
) -> list[Q]:
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


def translate_polymorphic_filter_definitions_in_args(
    queryset_model: type[models.Model], args: tuple[Q, ...], using: str = DEFAULT_DB_ALIAS
) -> list[Q]:
    return [translate_polymorphic_Q_object(queryset_model, q, using=using) for q in args]


def translate_polymorphic_Q_object(
    queryset_model: type[models.Model], potential_q_object: Q, using: str = DEFAULT_DB_ALIAS
) -> Q:
    if isinstance(potential_q_object, Q):
        return _translate_Q_tree(queryset_model, potential_q_object, using=using)
    return potential_q_object  # type: ignore[unreachable]


def _translate_Q_tree(
    queryset_model: type[models.Model],
    node: Q,
    using: str = DEFAULT_DB_ALIAS,
) -> Q:
    new_children: list[Any] = []
    for child in node.children:
        if isinstance(child, (tuple, list)):
            key, val = child
            new_expr = _translate_polymorphic_filter_definition(
                queryset_model, key, val, using=using
            )
            new_children.append(new_expr or child)
        elif isinstance(child, Q):
            new_children.append(
                _translate_Q_tree(queryset_model, child, using=using)
            )
        else:
            new_children.append(child)

    new_node = Q.__new__(Q)
    new_node.connector = node.connector
    new_node.negated = node.negated
    new_node.children = new_children
    return new_node


def _translate_polymorphic_filter_definition(
    queryset_model: type[models.Model],
    field_path: str,
    field_val: Any,
    using: str = DEFAULT_DB_ALIAS,
) -> tuple[str, Any] | Q | None:
    if field_path == "instance_of":
        return create_instanceof_q(field_val, using=using)
    elif field_path == "not_instance_of":
        return create_instanceof_q(field_val, not_instance_of=True, using=using)
    elif "___" not in field_path:
        return None

    newpath = translate_polymorphic_field_path(queryset_model, field_path)
    return (newpath, field_val)


def translate_polymorphic_field_path(queryset_model: type[models.Model], field_path: str) -> str:
    classname, sep, pure_field_path = field_path.partition("___")
    if not sep or not classname:
        return field_path

    negated = False
    if classname[0] == "-":
        negated = True
        classname = classname.lstrip("-")

    if "__" in classname:
        appname, sep, classname = classname.partition("__")
        try:
            model = apps.get_model(appname, classname)
        except LookupError as le:
            raise FieldError(f"Model {appname}.{classname} does not exist") from le
        if not issubclass(model, queryset_model):
            raise FieldError(
                f"{model._meta.label} is not derived from {queryset_model._meta.label}"
            )

    else:
        try:
            field = queryset_model._meta.get_field(classname)

            if isinstance(field, (RelatedField, ForeignObjectRel)):
                return field_path
        except FieldDoesNotExist:
            pass

        model = _map_queryname_to_class(queryset_model, classname)

    basepath = _create_base_path(queryset_model, model)

    if negated:
        newpath = "-"
    else:
        newpath = ""

    newpath += basepath
    if basepath:
        newpath += "__"

    newpath += pure_field_path
    return newpath


def _create_base_path(baseclass: type[models.Model], myclass: type[models.Model]) -> str:
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

    return myclass.__name__.lower()


def create_instanceof_q(
    modellist: type[models.Model] | list[type[models.Model]] | tuple[type[models.Model], ...],
    not_instance_of: bool = False,
    using: str = DEFAULT_DB_ALIAS,
) -> Q | None:
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
