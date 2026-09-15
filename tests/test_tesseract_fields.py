import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from nid_ocr_lab.dashboard.filters import apply_filter
from nid_ocr_lab.engines.tesseract_fields import refine_nid_fields
from nid_ocr_lab.models import OCRResult
from nid_ocr_lab.parsers.nid_parser import NIDParser


def word(text, x, y, width, height=12, line=1, conf=0.9):
    return {
        "text": text,
        "confidence": conf,
        "bounding_box": {"x": x, "y": y, "width": width, "height": height},
        "line_key": [1, 1, 1, line],
    }


# Page pass over a 336x208 card that dropped the পিতা row (y ~ 80) entirely.
PAGE_WORDS = [
    word("নাম:", 11, 20, 32, line=1), word("weer", 67, 18, 61, line=1), word("ত্রিপুরা", 131, 18, 47, line=1),
    word("Name:", 12, 52, 41, line=2), word("Doniel", 70, 51, 50, line=2), word("Tripura", 124, 51, 60, line=2),
    word("মাতা:", 11, 112, 30, line=3), word("নিরাজিতা", 68, 111, 56, line=3), word("ত্রিপুরা", 130, 111, 38, line=3),
    word("Date", 13, 143, 35, line=4), word("of", 53, 143, 15, line=4), word("Birth:", 72, 141, 38, line=4),
    word("28", 116, 142, 18, line=4, conf=0.95), word("May", 139, 141, 31, line=4, conf=0.95),
    word("2001", 175, 141, 34, line=4, conf=0.95),
    word("ID", 14, 167, 15, line=5), word("NO:", 35, 166, 28, line=5), word("8263067046", 69, 165, 93, line=5),
]

# (field, psm) -> (text, confidence). A missing PSM 13 entry means the retry must not happen.
REREADS = {
    ("field_name_bangla", 7): ("ওরেম", 0.3),  # weak first read ...
    ("field_name_bangla", 13): ("ডনিয়েল ত্রিপুরা", 0.8),  # ... fixed by the retry
    ("field_name_english", 7): ("Doniel Tripura", 0.8),
    ("field_father_name_bangla", 7): ("মসাধন ত্রিপুরা 7 |", 0.8),  # border speck read as a digit
    ("field_mother_name_bangla", 7): ("Some fig", 0.9),  # Latin junk on a Bengali row
    ("field_mother_name_bangla", 13): ("orem", 0.9),
    ("field_date_of_birth", 7): ("28 Way 2001", 0.8),  # page pass read it better
    ("field_nid_number", 7): ("8263067046", 0.9),
}


class FakeEngine:
    def __init__(self):
        self.calls = []

    def recognize(self, image_path, languages, preprocessing=None, psm=None, **kwargs):
        with Image.open(image_path) as crop:
            size = crop.size
        self.calls.append({"field": preprocessing, "languages": languages, "psm": psm, "size": size, **kwargs})
        text, confidence = REREADS.get((preprocessing, psm), ("", None))
        return OCRResult(
            blocks=[],
            full_text=text,
            engine="tesseract",
            latency_ms=10.0,
            metadata={"words": [{"text": text, "confidence": confidence}] if text else []},
        )


class NIDFieldRefinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name)
        self.image = self.workdir / "card.png"
        Image.new("RGB", (336, 208), "white").save(self.image)
        self.page = OCRResult(
            blocks=[],
            full_text="page text",
            engine="tesseract",
            language="ben+eng",
            preprocessing="as_uploaded_original_rot0_psm6",
            latency_ms=100.0,
            metadata={"words": PAGE_WORDS, "psm": 6},
        )
        self.engine = FakeEngine()
        self.result = refine_nid_fields(self.engine, self.image, self.page, variant="best", workdir=self.workdir)
        self.fields = {field["field"]: field for field in self.result.metadata["fields"]}

    def test_dropped_father_row_is_interpolated_and_read_in_bengali(self):
        father = self.fields["father_name_bangla"]
        self.assertEqual(father["source"], "interpolated")
        self.assertEqual(father["text"], "মসাধন ত্রিপুরা")
        # between the Name row (top 51) and the মাতা row (top 111)
        self.assertGreater(father["box"]["y"], 51)
        self.assertLess(father["box"]["y"] + father["box"]["height"], 111 + 12)
        call = next(call for call in self.engine.calls if call["field"] == "field_father_name_bangla")
        self.assertEqual((call["languages"], call["psm"], call["variant"]), (["ben"], 7, "best"))

    def test_each_row_is_locked_to_one_script(self):
        languages = {call["field"]: call["languages"] for call in self.engine.calls}
        self.assertEqual(languages["field_name_bangla"], ["ben"])
        self.assertEqual(languages["field_name_english"], ["eng"])
        # crops are upscaled to at least 64 px before the white border
        self.assertTrue(all(call["size"][1] >= 64 for call in self.engine.calls))

    def test_weak_read_is_retried_once_without_border(self):
        name_calls = [call for call in self.engine.calls if call["field"] == "field_name_bangla"]
        self.assertEqual([call["psm"] for call in name_calls], [7, 13])
        self.assertEqual(name_calls[0]["size"][1] - name_calls[1]["size"][1], 24)  # 12 px border on PSM 7 only
        self.assertEqual(self.fields["name_bangla"]["text"], "ডনিয়েল ত্রিপুরা")
        self.assertEqual(self.fields["name_bangla"]["source"], "reread")
        # retries: name_bangla (weak) and mother (wrong script); every other row stops after one read
        self.assertEqual(self.result.metadata["field_calls"], 8)
        self.assertAlmostEqual(self.result.latency_ms, 180.0)

    def test_wrong_script_reread_keeps_page_value(self):
        mother = self.fields["mother_name_bangla"]
        self.assertEqual(mother["source"], "page")
        self.assertEqual(mother["text"], "নিরাজিতা ত্রিপুরা")
        self.assertEqual([item["score"] for item in mother["candidates"][1:]], [0.0, 0.0])

    def test_confident_page_value_beats_weaker_reread(self):
        dob = self.fields["date_of_birth"]
        self.assertEqual((dob["source"], dob["text"]), ("page", "28 May 2001"))

    def test_nid_row_uses_digit_whitelist(self):
        call = next(call for call in self.engine.calls if call["field"] == "field_nid_number")
        self.assertEqual(call["config"], {"tessedit_char_whitelist": "0123456789"})
        others = [call["config"] for call in self.engine.calls if call["field"] != "field_nid_number"]
        self.assertTrue(all(config is None for config in others))

    def test_refined_text_parses_into_fields(self):
        self.assertTrue(self.result.preprocessing.endswith("_fields"))
        self.assertEqual(self.result.metadata["page_full_text"], "page text")
        parsed = NIDParser().parse(self.result)
        self.assertEqual(parsed.father_name_bangla.raw_value, "মসাধন ত্রিপুরা")
        self.assertEqual(parsed.name_bangla.raw_value, "ডনিয়েল ত্রিপুরা")
        self.assertEqual(parsed.name_english.raw_value, "Doniel Tripura")
        self.assertEqual(parsed.mother_name_bangla.raw_value, "নিরাজিতা ত্রিপুরা")
        self.assertEqual(parsed.nid_number.raw_value, "8263067046")
        self.assertEqual(parsed.date_of_birth.raw_value, "2001-05-28")
        # blocks hold only the value text, in card order, so label-panel crops match their ground truth
        self.assertEqual(self.result.blocks[2].text, "মসাধন ত্রিপুরা")

    def test_page_value_keeps_tesseract_word_order(self):
        # Tesseract's reading order wins over slightly overlapping x boxes
        words = [dict(item) for item in PAGE_WORDS]
        may = next(index for index, item in enumerate(words) if item["text"] == "May")
        words[may] = word("May", 110, 141, 31, line=4, conf=0.95)
        page = OCRResult(blocks=[], full_text="", engine="tesseract", metadata={"words": words})
        result = refine_nid_fields(FakeEngine(), self.image, page, variant="best", workdir=self.workdir)
        dob = next(field for field in result.metadata["fields"] if field["field"] == "date_of_birth")
        self.assertEqual(dob["text"], "28 May 2001")

    def test_unmatched_page_lines_are_kept(self):
        words = [word("গণপ্রজাতন্ত্রী", 40, 2, 200, line=9)] + PAGE_WORDS
        page = OCRResult(blocks=[], full_text="", engine="tesseract", metadata={"words": words})
        result = refine_nid_fields(FakeEngine(), self.image, page, variant="best", workdir=self.workdir)
        self.assertEqual(result.full_text.splitlines()[0], "গণপ্রজাতন্ত্রী")


class NIDInkFilterTests(unittest.TestCase):
    def test_emblem_yellow_turns_white_and_ink_stays_dark(self):
        image = Image.new("RGB", (4, 1))
        for x, color in enumerate([(20, 20, 20), (200, 30, 40), (235, 200, 60), (250, 250, 250)]):
            image.putpixel((x, 0), color)
        output = apply_filter(image, "nid_ink").convert("L")
        black, red, yellow, paper = (output.getpixel((x, 0)) for x in range(4))
        self.assertLess(black, 60)
        self.assertLess(red, 60)
        self.assertGreater(yellow, 180)
        self.assertGreater(paper, 200)


if __name__ == "__main__":
    unittest.main()
