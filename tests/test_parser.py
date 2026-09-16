import unittest

from nid_ocr_lab.evaluation import character_error_rate, evaluate_fields, summarize
from nid_ocr_lab.models import OCRResult
from nid_ocr_lab.parsers.nid_parser import NIDParser

SMART_NID_TEXT = """গণপ্রজাতন্ত্রী বাংলাদেশ সরকার
Government of the People's Republic of Bangladesh
National ID Card / জাতীয় পরিচয় পত্র
নাম: রহিম উদ্দিন
Name: RAHIM UDDIN
পিতা: করিম উদ্দিন
মাতা: আমিনা বেগম
Date of Birth: 31 Jan 1990
ID NO: ১২৩৪৫৬৭৮৯০"""


def parse(text):
    return NIDParser().parse(OCRResult(blocks=[], full_text=text, engine="test"))


class ParserTests(unittest.TestCase):
    def test_labels_route_by_script(self):
        result = parse(SMART_NID_TEXT)
        self.assertEqual(result.name_bangla.value, "রহিম উদ্দিন")
        self.assertEqual(result.name_english.value, "RAHIM UDDIN")
        self.assertEqual(result.father_name_bangla.value, "করিম উদ্দিন")
        self.assertEqual(result.mother_name_bangla.value, "আমিনা বেগম")
        self.assertIsNone(result.father_name_english.value)
        self.assertEqual(result.date_of_birth.value, "1990-01-31")
        self.assertEqual(result.nid_number.value, "1234567890")

    def test_relation_name_label_does_not_fill_holder_name(self):
        result = parse("Father's Name: KARIM UDDIN\nMother's Name: AMINA BEGUM")
        self.assertIsNone(result.name_english.value)
        self.assertEqual(result.father_name_english.value, "KARIM UDDIN")
        self.assertEqual(result.mother_name_english.value, "AMINA BEGUM")

    def test_value_on_next_line_and_script_mismatch(self):
        result = parse("নাম:\nরহিম উদ্দিন\nName: রহিম")
        self.assertEqual(result.name_bangla.value, "রহিম উদ্দিন")
        self.assertEqual(result.name_bangla.source, "label-next-line")
        self.assertIsNone(result.name_english.value)
        self.assertTrue(result.name_english.source.endswith("script-mismatch"))

    def test_dates_must_be_real_and_iso(self):
        self.assertIsNone(parse("Date of Birth: 31 Feb 1990").date_of_birth.value)
        self.assertEqual(parse("Date of Birth: 05/11/1985").date_of_birth.value, "1985-11-05")
        self.assertIsNone(parse("Date of Bir h: 28 May 001").date_of_birth.value)


class EvaluationTests(unittest.TestCase):
    def test_unlabeled_fields_are_not_counted(self):
        metrics = evaluate_fields(parse(SMART_NID_TEXT), {"name_english": "Rahim Uddin", "nid_number": "1234567899"})
        summary = summarize(metrics)
        self.assertEqual(summary["fields_total"], 2)
        self.assertEqual(summary["fields_correct"], 1)
        self.assertEqual(summary["fields_unlabeled"], 8)
        self.assertEqual(summary["per_script"]["english"]["exact_accuracy"], 1.0)
        self.assertAlmostEqual(summary["per_field"]["nid_number"]["mean_cer"], 0.1)

    def test_cer(self):
        self.assertEqual(character_error_rate("abc", "abc"), 0.0)
        self.assertAlmostEqual(character_error_rate("abd", "abc"), 1 / 3)


if __name__ == "__main__":
    unittest.main()


class SpacedNIDNumberTests(unittest.TestCase):
    def test_number_printed_in_groups(self):
        from nid_ocr_lab.parsers.nid_parser import find_nid_number

        self.assertEqual(find_nid_number("NID No: 597 029 5035").raw_value, "5970295035")
        self.assertEqual(find_nid_number("ID NO: 8713389347").raw_value, "8713389347")
        # a misread label still leaves the digits findable
        self.assertEqual(find_nid_number("(৮9. nove 597 029 5035").raw_value, "5970295035")
        # short or date-like runs are not NID numbers
        self.assertIsNone(find_nid_number("Date of Birth 10 Dec 1989").raw_value)
        self.assertIsNone(find_nid_number("NID No: 597").raw_value)
