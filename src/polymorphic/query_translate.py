"""
PolymorphicQuerySet support functions - Refactored QueryTranslate mechanism

This module provides a new Q object parser that supports:
- Q & Q (AND)
- Q | Q (OR)
- Q ^ Q (XOR)
- ~Q (NOT)
- Nested combinations of the above
- Cross-inheritance-level field filtering using the '___' syntax
- instance_of and not_instance_of expressions
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from operator import or_
from typing import Any, Callable, Optional

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import models
from django.db.models import Q, Subquery
from django.db.models.fields.related import ForeignObjectRel, RelatedField
from django.db.utils import DEFAULT_DB_ALIAS

from .utils import _lazy_ctype, _map_queryname_to_class, concrete_descendants


class QOperator(Enum):
    """Q object logical operators"""
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"


@dataclass
class QLookup:
    """Represents a single lookup expression in a Q object"""
    lookup: str
    value: Any


@dataclass
class QNode(ABC):
    """Abstract base class for Q expression tree nodes"""
    @abstractmethod
    def to_django_q(self) -> Q:
        """Convert this node to a Django Q object"""
        pass


@dataclass
class QLeafNode(QNode):
    """A leaf node representing a single lookup expression"""
    lookup: str
    value: Any
    translated: bool = False
    translated_lookup: Optional[str] = None
    translated_value: Any = None
    q_object: Optional[Q] = None

    def translate(self, queryset_model: type[models.Model], using: str) -> None:
        """Translate the lookup if it contains polymorphic syntax"""
        if self.translated:
            return

        if self.lookup == "instance_of":
            self.q_object = create_instanceof_q(self.value, using=using)
            self.translated = True
            return

        if self.lookup == "not_instance_of":
            self.q_object = create_instanceof_q(self.value, not_instance_of=True, using=using)
            self.translated = True
            return

        if "___" not in self.lookup:
            self.translated = True
            self.translated_lookup = self.lookup
            self.translated_value = self.value
            return

        self.translated_lookup = translate_polymorphic_field_path(queryset_model, self.lookup)
        self.translated_value = self.value
        self.translated = True

    def to_django_q(self) -> Q:
        if not self.translated:
            raise ValueError("QLeafNode must be translated before conversion")

        if self.q_object is not None:
            return self.q_object

        return Q(**{self.translated_lookup: self.translated_value})


@dataclass
class QBranchNode(QNode):
    """A branch node representing a logical operation on child nodes"""
    operator: QOperator
    children: list[QNode] = field(default_factory=list)
    negated: bool = False

    def add_child(self, child: QNode) -> None:
        self.children.append(child)

    def to_django_q(self) -> Q:
        if not self.children:
            return Q()

        result = self._combine_children()

        if self.negated:
            return ~result

        return result

    def _combine_children(self) -> Q:
        if len(self.children) == 1:
            return self.children[0].to_django_q()

        q_objects = [child.to_django_q() for child in self.children]

        if self.operator == QOperator.AND:
            return reduce(lambda a, b: a & b, q_objects)
        elif self.operator == QOperator.OR:
            return reduce(lambda a, b: a | b, q_objects)
        elif self.operator == QOperator.XOR:
            return reduce(lambda a, b: a ^ b, q_objects)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class PolymorphicQParser:
    """
    Parser for polymorphic Q objects.

    Traverses a Django Q object tree and translates polymorphic expressions
    (like 'ModelX___field' and 'instance_of') into standard Django filter expressions.

    Supports all Q object operations:
    - Q & Q (AND)
    - Q | Q (OR)
    - Q ^ Q (XOR)
    - ~Q (NOT)
    - Nested combinations of the above
    """

    def __init__(self, queryset_model: type[models.Model], using: str = DEFAULT_DB_ALIAS):
        self.queryset_model = queryset_model
        self.using = using

    def parse(self, q_object: Q) -> Q:
        """
        Parse and translate a Q object tree.

        Args:
            q_object: The Q object to parse

        Returns:
            A new Q object with all polymorphic expressions translated
        """
        if not isinstance(q_object, Q):
            return q_object

        tree = self._build_tree(q_object)
        self._translate_tree(tree)
        return tree.to_django_q()

    def _build_tree(self, q_object: Q, parent_negated: bool = False) -> QNode:
        """Build an expression tree from a Django Q object"""
        operator = self._get_operator(q_object)
        branch = QBranchNode(operator=operator, negated=q_object.negated)

        for child in q_object.children:
            if isinstance(child, Q):
                branch.add_child(self._build_tree(child))
            elif isinstance(child, (tuple, list)) and len(child) == 2:
                lookup, value = child
                leaf = QLeafNode(lookup=lookup, value=value)
                branch.add_child(leaf)
            else:
                branch.add_child(QLeafNode(lookup=str(child), value=True))

        return branch

    def _translate_tree(self, node: QNode) -> None:
        """Recursively translate all leaf nodes in the tree"""
        if isinstance(node, QLeafNode):
            node.translate(self.queryset_model, self.using)
        elif isinstance(node, QBranchNode):
            for child in node.children:
                self._translate_tree(child)

    def _get_operator(self, q_object: Q) -> QOperator:
        """Determine the logical operator for a Q object based on its connector"""
        connector = q_object.connector
        if connector == Q.AND:
            return QOperator.AND
        elif connector == Q.OR:
            return QOperator.OR
        elif connector == Q.XOR:
            return QOperator.XOR
        else:
            return QOperator.AND


def translate_polymorphic_Q_object(
    queryset_model: type[models.Model], potential_q_object: Q, using: str = DEFAULT_DB_ALIAS
) -> Q:
    """
    Translate a Q object with polymorphic expressions into a standard Django Q object.

    This is the main entry point for Q object translation. It uses the new
    PolymorphicQParser to handle all Q object operations including:
    - Q & Q (AND)
    - Q | Q (OR)
    - Q ^ Q (XOR)
    - ~Q (NOT)
    - Nested combinations

    Args:
        queryset_model: The base model of the queryset
        potential_q_object: The Q object to translate
        using: Database alias

    Returns:
        A translated Q object with standard Django filter expressions
    """
    if not isinstance(potential_q_object, models.Q):
        return potential_q_object

    parser = PolymorphicQParser(queryset_model, using=using)
    return parser.parse(potential_q_object)


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
    if field_path == "instance_of":
        return create_instanceof_q(field_val, using=using)
    elif field_path == "not_instance_of":
        return create_instanceof_q(field_val, not_instance_of=True, using=using)
    elif "___" not in field_path:
        return None

    newpath = translate_polymorphic_field_path(queryset_model, field_path)
    return (newpath, field_val)


def translate_polymorphic_field_path(queryset_model: type[models.Model], field_path: str) -> str:
    """
    Translate a field path from a keyword argument, as used for
    PolymorphicQuerySet.filter()-like functions (and Q objects).
    Supports leading '-' (for order_by args).

    E.g.: if queryset_model is ModelA, then "ModelC___field3" is translated
    into modela__modelb__modelc__field3.

    Supports cross-inheritance-level field filtering by:
    1. Resolving the target model class from the path prefix
    2. Building the proper join path through the inheritance hierarchy
    3. Appending the actual field path

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
    """
    Create the field path for polymorphic expressions through the inheritance hierarchy.

    Traverses the inheritance chain from myclass up to baseclass, building
    the proper Django ORM join path.

    For example, for baseclass=ModelA, myclass=ModelC (where ModelC inherits from
    ModelB which inherits from ModelA), returns 'modelb__modelc'.
    """
    for b in myclass.__bases__:
        if b == baseclass:
            return _get_query_related_name(myclass)

        path = _create_base_path(baseclass, b)
        if path:
            if b._meta.abstract or b._meta.proxy:
                return _get_query_related_name(myclass)
            else:
                return f"{path}__{_get_query_related_name(myclass)}"
    return ""


def _get_query_related_name(myclass: type[models.Model]) -> str:
    """Get the related query name for a model's parent link field"""
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
    """Get content type IDs for models and their concrete descendants"""
    lazy: list[Q] = []
    ids: list[int] = []
    for model in models:
        cid = _lazy_ctype(model, using=using)
        ids.append(cid.pk) if isinstance(cid, ContentType) else lazy.append(cid)
        for descendent in concrete_descendants(model, include_proxy=True):
            cid = _lazy_ctype(descendent, using=using)
            ids.append(cid.pk) if isinstance(cid, ContentType) else lazy.append(cid)
    return lazy, ids
