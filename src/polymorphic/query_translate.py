"""
PolymorphicQuerySet support functions

Provides a refactored mechanism for translating polymorphic
filter definitions and Q objects into queries that Django
understands.

Design
------

The polymorphic query translate pipeline has two independent
facilities:

1. :class:`PolymorphicFieldPathTranslator` - translates a
   "ClassName___field" style field path to a regular Django
   field path (e.g. "modela__modelb__modelc__field").
2. :class:`PolymorphicQTranslator` - recursively walks a
   :class:`django.db.models.Q` expression tree (supporting
   ``Q & Q``, ``Q | Q``, ``Q ^ Q`` and arbitrary nesting)
   and rewrites any polymorphic constructs it encounters.

On top of these two primitives we keep the thin helper
functions required by :class:`~polymorphic.query.PolymorphicQuerySet`
so the existing public API of this module is preserved.
"""

from __future__ import annotations

import copy
from functools import reduce
from operator import or_
from typing import Any, Iterable, List, Tuple, Union, cast

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import models
from django.db.models import Q, Subquery
from django.db.models.fields.related import ForeignObjectRel, RelatedField
from django.db.utils import DEFAULT_DB_ALIAS

from .utils import _lazy_ctype, _map_queryname_to_class, concrete_descendants

# ---------------------------------------------------------------------------
# Public API - thin wrappers used by PolymorphicQuerySet
# ---------------------------------------------------------------------------


def translate_polymorphic_filter_definitions_in_kwargs(
    queryset_model: type[models.Model],
    kwargs: dict[str, Any],
    using: str = DEFAULT_DB_ALIAS,
) -> list[Q]:
    """
    Translate the keyword argument list for ``PolymorphicQuerySet.filter()``.

    Any kwargs that contain special polymorphic functionality (``instance_of``,
    ``not_instance_of``, ``ClassName___field``) are replaced in-place in the
    *kwargs* dict with their vanilla Django equivalents.  Filters that can
    only be expressed with a :class:`Q` object are removed from *kwargs*
    and appended to the returned list.
    """
    additional_args: list[Q] = []
    translator = _PolymorphicFilterTranslator(queryset_model, using=using)

    # iterate over a snapshot so we can mutate ``kwargs`` safely
    for field_path, val in list(kwargs.items()):
        result = translator.translate(field_path, val)

        if isinstance(result, Q):
            del kwargs[field_path]
            additional_args.append(result)
        elif isinstance(result, tuple):
            new_path, new_val = result
            if new_path != field_path or new_val is not val:
                del kwargs[field_path]
                kwargs[new_path] = new_val

    return additional_args


def translate_polymorphic_Q_object(
    queryset_model: type[models.Model],
    q: Q,
    using: str = DEFAULT_DB_ALIAS,
) -> Q:
    """Translate a :class:`Q` tree, resolving any polymorphic constructs."""
    return PolymorphicQTranslator(queryset_model, using=using).translate(q)


def translate_polymorphic_filter_definitions_in_args(
    queryset_model: type[models.Model],
    args: tuple[Q, ...] | Iterable[Q],
    using: str = DEFAULT_DB_ALIAS,
) -> list[Q]:
    """
    Translate each top-level :class:`Q` argument passed to a polymorphic
    filter/exclude call.  The returned list is the translated equivalent
    suitable to be forwarded to Django's regular ``_filter_or_exclude``.
    """
    return [translate_polymorphic_Q_object(queryset_model, arg, using=using) for arg in args]


def translate_polymorphic_field_path(
    queryset_model: type[models.Model], field_path: str
) -> str:
    """
    Translate a ``ClassName___field`` style field path into a vanilla
    Django path (``parent__child__field``).  Supports leading ``-``
    (used by ``order_by`` / ``defer`` / ``only``).
    """
    return PolymorphicFieldPathTranslator(queryset_model).translate(field_path)


def create_instanceof_q(
    modellist: (
        type[models.Model]
        | list[type[models.Model]]
        | tuple[type[models.Model], ...]
        | None
    ),
    not_instance_of: bool = False,
    using: str = DEFAULT_DB_ALIAS,
) -> Q | None:
    """
    Build a :class:`Q` that filters rows whose ``polymorphic_ctype`` refers
    to one of the models in *modellist* (or any of their concrete
    descendants).  The returned query is lazily evaluated so it is safe to
    call this function before Django's apps are fully loaded.
    """
    if not modellist:
        return None

    # allow a single class to be passed directly
    if not isinstance(modellist, (list, tuple)):
        from .models import PolymorphicModel

        if not issubclass(modellist, PolymorphicModel):
            raise TypeError(
                "PolymorphicModel: instance_of expects a list of (polymorphic) "
                "models or a single (polymorphic) model"
            )
        modellist = [modellist]

    lazy_cts, ct_ids = _collect_content_types(modellist, using)

    # Build a single Q that matches rows whose polymorphic_ctype is in the
    # union of the given content types.  Using separate Subquery-based and
    # id-based clauses keeps the query plan simple.
    q = Q()
    if lazy_cts:
        q |= Q(
            polymorphic_ctype__in=Subquery(
                ContentType.objects.filter(reduce(or_, lazy_cts)).values("pk")
            )
        )
    if ct_ids:
        q |= Q(polymorphic_ctype__in=list(ct_ids))

    if not_instance_of:
        q = ~q
    return q


# ---------------------------------------------------------------------------
# Core field path translator
# ---------------------------------------------------------------------------


class PolymorphicFieldPathTranslator:
    """
    Translate ``App__Model___field`` style filter expressions into the
    corresponding multi-table join paths Django's ORM expects.

    The class exposes a single :meth:`translate` entry point.  Translators
    are stateless and safe to reuse across multiple field paths for the
    same base model.
    """

    __slots__ = ("base_model",)

    def __init__(self, base_model: type[models.Model]) -> None:
        self.base_model = base_model

    # -- public API -----------------------------------------------------

    def translate(self, field_path: str) -> str:
        """Return a translated field path, or the original if no translation
        is required."""
        if not isinstance(field_path, str) or "___" not in field_path:
            return field_path

        negated, raw = self._strip_negation(field_path)
        class_spec, _, rest = raw.partition("___")

        # Allow the user to specify an app label via ``App__Model___field``.
        if "__" in class_spec:
            app_name, _, class_name = class_spec.partition("__")
            model = apps.get_model(app_name, class_name)
            if not issubclass(model, self.base_model):
                raise FieldError(
                    f"{model._meta.label} is not derived from "
                    f"{self.base_model._meta.label}"
                )
        else:
            # Check whether this is simply a regular field on the base model.
            # We only fall back to polymorphic name resolution when no such
            # field exists.
            if self._is_regular_field(class_spec):
                return field_path

            try:
                model = _map_queryname_to_class(self.base_model, class_spec)
            except AssertionError as exc:
                raise FieldError(str(exc)) from exc

        base_path = _create_base_path(self.base_model, model)
        if base_path:
            prefix = base_path + "__"
        else:
            prefix = ""

        return ("-" if negated else "") + prefix + rest

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _strip_negation(field_path: str) -> Tuple[bool, str]:
        if field_path.startswith("-"):
            return True, field_path.lstrip("-")
        return False, field_path

    def _is_regular_field(self, name: str) -> bool:
        """Return True if *name* refers to a real field on the base model
        (which means no polymorphic expansion is desired)."""
        try:
            field = self.base_model._meta.get_field(name)
        except FieldDoesNotExist:
            return False
        return isinstance(field, (RelatedField, ForeignObjectRel))


# ---------------------------------------------------------------------------
# Recursive Q object translator
# ---------------------------------------------------------------------------


class PolymorphicQTranslator:
    """
    Walk a :class:`django.db.models.Q` expression tree and rewrite every
    polymorphic construct it encounters.

    Supported features
    ------------------

    * Arbitrary boolean connectors (``AND`` / ``OR`` / ``XOR``) and
      negation, at any nesting depth.
    * ``instance_of=...`` and ``not_instance_of=...`` lookups.
    * ``ClassName___field`` style field paths inside ``Q`` leaves, at any
      depth of the tree.
    * Children that are themselves ``Q`` instances (nested expressions).

    The translator creates a fresh ``Q`` tree so callers never observe
    mutation of the original expression.
    """

    __slots__ = ("base_model", "using", "_field_path_translator")

    def __init__(self, base_model: type[models.Model], using: str = DEFAULT_DB_ALIAS) -> None:
        self.base_model = base_model
        self.using = using
        self._field_path_translator = PolymorphicFieldPathTranslator(base_model)

    # -- public API -----------------------------------------------------

    def translate(self, q: Q) -> Q:
        """Return a translated copy of *q*.  Non-:class:`Q` objects are
        returned unmodified."""
        if not isinstance(q, Q):
            return q
        return self._translate_node(q)

    # -- recursive traversal -------------------------------------------

    def _translate_node(self, node: Q) -> Q:
        # Shallow-copy the node; we replace its children entirely so we do
        # not mutate the caller's Q object.
        new_node = copy.copy(node)
        new_node.children = []

        for child in node.children:
            new_node.children.append(self._translate_child(child))

        return new_node

    def _translate_child(self, child: Any) -> Any:
        if isinstance(child, Q):
            return self._translate_node(child)
        if isinstance(child, (tuple, list)) and len(child) == 2:
            key, val = child[0], child[1]
            return self._translate_leaf(key, val)
        # Fall-through: unknown shape; leave it alone and let Django handle
        # the object as-is.
        return child

    def _translate_leaf(self, key: Any, val: Any) -> Union[Q, Tuple[str, Any]]:
        # Only string keys are subject to polymorphic rewriting.
        if not isinstance(key, str):
            return (key, val)

        if key == "instance_of":
            q = create_instanceof_q(val, using=self.using)
            return q if q is not None else (key, val)
        if key == "not_instance_of":
            q = create_instanceof_q(val, not_instance_of=True, using=self.using)
            return q if q is not None else (key, val)

        new_key = self._field_path_translator.translate(key)
        return (new_key, val)


# ---------------------------------------------------------------------------
# Internal helpers used by the translators
# ---------------------------------------------------------------------------


class _PolymorphicFilterTranslator:
    """Internal helper; wraps a :class:`PolymorphicQTranslator` plus
    :meth:`create_instanceof_q` and :class:`PolymorphicFieldPathTranslator`
    into a single ``(path, value) -> Q | tuple | None`` callable used by
    the kwargs translation pipeline."""

    __slots__ = ("_q_translator", "_field_path_translator", "_using")

    def __init__(self, base_model: type[models.Model], using: str = DEFAULT_DB_ALIAS) -> None:
        self._q_translator = PolymorphicQTranslator(base_model, using=using)
        self._field_path_translator = PolymorphicFieldPathTranslator(base_model)
        self._using = using

    def translate(self, key: str, value: Any) -> Union[Q, Tuple[str, Any], None]:
        if key == "instance_of":
            return create_instanceof_q(value, using=self._using)
        if key == "not_instance_of":
            return create_instanceof_q(value, not_instance_of=True, using=self._using)

        # Q object values should already be processed by the args pipeline;
        # keep a safety net that walks them if one arrives here.
        if isinstance(value, Q):
            return self._q_translator.translate(Q(**{key: value}))

        new_key = self._field_path_translator.translate(key)
        if new_key == key:
            return None
        return (new_key, value)


def _create_base_path(baseclass: type[models.Model], myclass: type[models.Model]) -> str:
    """
    Compute the join path from *baseclass* to *myclass* using the
    ``OneToOneField(parent_link=True)`` Django implicitly creates for
    multi-table inheritance.  Abstract/proxy models do not contribute
    a table so they are skipped.
    """
    if myclass is baseclass:
        return ""

    for parent in myclass.__bases__:
        if not hasattr(parent, "_meta"):
            # skip non-model mixins
            continue
        if parent is baseclass:
            return _get_query_related_name(myclass)

        path = _create_base_path(baseclass, parent)
        if path:
            if parent._meta.abstract or parent._meta.proxy:  # type: ignore[attr-defined]
                return _get_query_related_name(myclass)
            return f"{path}__{_get_query_related_name(myclass)}"
    return ""


def _get_query_related_name(myclass: type[models.Model]) -> str:
    """Return the query name Django uses for a reverse join to *myclass*."""
    for field in myclass._meta.local_fields:
        if isinstance(field, models.OneToOneField) and field.remote_field.parent_link:
            return field.related_query_name()

    # Fallback used for proxy models, which never own their own OneToOne
    # parent link.  Using the bare lower-cased class name mirrors the
    # behaviour the public API has always exposed.
    return myclass.__name__.lower()


def _collect_content_types(
    modellist: Iterable[type[models.Model]], using: str
) -> Tuple[List[Q], List[int]]:
    """Collect the content types for the given models and all of their
    concrete descendants, returning (lazy_Q_filters, resolved_ids)."""
    lazy: list[Q] = []
    ids: list[int] = []
    for model in modellist:
        cid = _lazy_ctype(model, using=using)
        if isinstance(cid, ContentType):
            ids.append(cid.pk)
        else:
            lazy.append(cast(Q, cid))

        for descendant in concrete_descendants(model, include_proxy=True):
            cid = _lazy_ctype(descendant, using=using)
            if isinstance(cid, ContentType):
                ids.append(cid.pk)
            else:
                lazy.append(cast(Q, cid))

    return lazy, ids


# Backwards-compatible alias for the private helper still used by
# :func:`create_instanceof_q` above.  Keeps the symbol visible for any
# third-party code that might have relied on it.
_get_mro_content_type_ids = _collect_content_types
