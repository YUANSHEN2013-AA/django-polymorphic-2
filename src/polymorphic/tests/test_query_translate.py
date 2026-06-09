import copy
import pickle
import tempfile
import threading

from django.db.models import Q
from django.test import TestCase

from polymorphic.query_translate import (
    translate_polymorphic_Q_object,
    translate_polymorphic_field_path,
    translate_polymorphic_filter_definitions_in_args,
)
from polymorphic.tests.models import Bottom, DeepCopyTester, DeepCopyTester2, Model2A, Model2B, Model2C, Model2D


class QueryTranslateTests(TestCase):
    def create_model2_chain_with_overlap(self):
        a = Model2A.objects.create(field1="shared")
        b = Model2B.objects.create(field1="shared", field2="match-b")
        c = Model2C.objects.create(field1="shared", field2="match-c", field3="match-c3")
        d = Model2D.objects.create(field1="shared", field2="match-b", field3="match-c3", field4="match-d4")
        return a, b, c, d

    def test_translate_with_not_pickleable_query(self):
        with tempfile.TemporaryFile() as fd:
            with self.assertRaises(TypeError):
                pickle.dumps(threading.Lock())

            q = Q(blog__info="blog info") | Q(blog__info=threading.Lock())

            translate_polymorphic_filter_definitions_in_args(Bottom, args=[q])

    def test_deep_copy_of_q_objects(self):
        import os

        d1_bf = os.urandom(32)
        d2_bf1 = os.urandom(32)
        d2_bf2 = os.urandom(32)

        dct1 = DeepCopyTester.objects.create(binary_field=d1_bf)
        dct2 = DeepCopyTester2.objects.create(binary_field=d2_bf1, binary_field2=d2_bf2)

        self.assertEqual(list(DeepCopyTester.objects.filter(binary_field=d1_bf).all()), [dct1])

        q1 = Q(DeepCopyTester2___binary_field2=d2_bf1)
        self.assertEqual(list(DeepCopyTester.objects.filter(q1).all()), [])
        assert q1.children[0][0] == "DeepCopyTester2___binary_field2"
        q2 = Q(DeepCopyTester2___binary_field2=d2_bf2)
        self.assertEqual(list(DeepCopyTester.objects.filter(q2).all()), [dct2])
        assert q2.children[0][0] == "DeepCopyTester2___binary_field2"

        assert len(DeepCopyTester.objects.filter(Q(binary_field=memoryview(d1_bf)))) == 1

        self.assertEqual(DeepCopyTester.objects.all().delete()[0], 3)
        self.assertEqual(DeepCopyTester.objects.count(), 0)

    def test_proxy_model_query_related_name(self):
        from polymorphic.query_translate import _get_query_related_name
        from polymorphic.tests.models import ProxyChild, SubclassSelectorProxyModel

        result = _get_query_related_name(ProxyChild)
        assert result == "proxychild"

        result = _get_query_related_name(SubclassSelectorProxyModel)
        assert result == "subclassselectorproxymodel"

    def test_translate_polymorphic_q_object_supports_nested_boolean_connectors(self):
        q_object = Q(Model2A___field1="shared") & (
            Q(Model2B___field2="match-b") ^ Q(Model2D___field4="match-d4")
        )
        original_q_object = copy.deepcopy(q_object)

        translated_q_object = translate_polymorphic_Q_object(Model2C, q_object)

        assert q_object.children == original_q_object.children
        assert translated_q_object.connector == Q.AND
        assert translated_q_object.children[0][0] == "field1"
        assert translated_q_object.children[1].connector == Q.XOR
        assert translated_q_object.children[1].children[0][0] == "field2"
        assert translated_q_object.children[1].children[1][0] == "model2d__field4"

    def test_translate_polymorphic_field_path_supports_cross_inheritance_levels(self):
        assert translate_polymorphic_field_path(Model2C, "Model2A___field1") == "field1"
        assert translate_polymorphic_field_path(Model2B, "Model2D___field4") == "model2c__model2d__field4"
        assert translate_polymorphic_field_path(Model2D, "tests__Model2B___field2") == "field2"

    def test_filter_supports_nested_boolean_q_expressions(self):
        _, _, _, d = self.create_model2_chain_with_overlap()

        results = list(
            Model2A.objects.filter(
                Q(Model2B___field2="match-b")
                & (Q(Model2C___field3="match-c3") | Q(Model2D___field4="match-d4"))
            ).order_by("pk")
        )

        assert results == [d]

    def test_filter_supports_xor_q_expressions(self):
        _, b, c, _ = self.create_model2_chain_with_overlap()

        results = list(
            Model2A.objects.filter(
                Q(Model2B___field2="match-b") ^ Q(Model2C___field3="match-c3")
            ).order_by("pk")
        )

        assert results == [b, c]

    def test_filter_supports_cross_inheritance_field_filters(self):
        _, _, c, d = self.create_model2_chain_with_overlap()

        results = list(
            Model2C.objects.filter(
                Q(Model2A___field1="shared")
                & (Q(Model2B___field2="match-c") | Q(Model2D___field4="match-d4"))
            ).order_by("pk")
        )

        assert results == [c, d]
