import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nid_ocr_lab.dashboard import server
from nid_ocr_lab.engines import tesseract_capi as capi
from nid_ocr_lab.engines import tesseract_steps as steps
from nid_ocr_lab.engines.tessdata import get_variant
from nid_ocr_lab.engines.tesseract import TesseractEngine


class ThresholdMathTests(unittest.TestCase):
    def test_otsu_splits_two_levels_and_marks_dark_ink(self):
        grey = np.full((40, 40), 220, np.uint8)
        grey[10:20, 5:30] = 30  # 25% dark ink on light paper
        channels, ink = steps.tesseract_otsu(grey[..., None])
        self.assertEqual(len(channels), 1)
        self.assertTrue(30 <= channels[0]["threshold"] < 220)
        self.assertEqual(channels[0]["ink"], f"≤ {channels[0]['threshold']} is ink")
        self.assertTrue(ink[15, 10])
        self.assertFalse(ink[0, 0])
        self.assertEqual(int(ink.sum()), 250)

    def test_flat_channel_is_ignored(self):
        rgb = np.full((20, 20, 3), 200, np.uint8)
        rgb[5:10, 5:10, 0] = 10  # only red varies
        channels, _ink = steps.tesseract_otsu(np.dstack([rgb, np.zeros((20, 20), np.uint8)]))
        used = [item["channel"] for item in channels if item["used"]]
        self.assertEqual(used, ["red"])
        self.assertEqual(channels[3]["ink"], "ignored")

    def test_window_sizes_follow_tesseract_formulas(self):
        self.assertEqual(steps.threshold_resolution(0, None), 70)
        self.assertEqual(steps.threshold_resolution(300, None), 300)
        self.assertEqual(steps.threshold_resolution(3000, None), 70)
        self.assertEqual(steps.threshold_resolution(0, 450), 450)
        self.assertEqual(steps.adaptive_otsu_sizes(70, 0.33, 0), {"tile": 23, "smooth": 0, "half_smooth": 0})
        self.assertEqual(steps.adaptive_otsu_sizes(30, 0.33, 0)["tile"], 16)  # never below 16
        sizes = steps.sauvola_sizes(70, 0.33, 1780, 1123)
        self.assertEqual((sizes["window"], sizes["half_window"], sizes["nx"], sizes["ny"]), (23, 11, 7, 4))

    def test_threshold_settings_are_validated(self):
        self.assertEqual(steps.normalize_threshold(None)["method"], "otsu")
        self.assertEqual(steps.normalize_threshold({"method": "manual", "value": "140.4"})["value"], 140)
        for bad in ({"method": "magic"}, {"method": "manual", "value": 300}, {"kfactor": -1}, {"tile_size": "x"}):
            with self.assertRaises(ValueError):
                steps.normalize_threshold(bad)

    def test_step_file_route_rejects_traversal(self):
        self.assertIsNotNone(server.STEP_FILE_PATTERN.match("/api/steps/20260915-120000-0123abcd/03-binary.png"))
        for path in ("/api/steps/20260915-120000-0123abcd/../x.png", "/api/steps/../../etc/passwd", "/api/steps/20260915-120000-0123abcd/a.txt"):
            self.assertIsNone(server.STEP_FILE_PATTERN.match(path))


def english_model_available() -> bool:
    try:
        return capi.is_available() and shutil.which("tesseract") is not None and get_variant("system").model_path("eng").is_file()
    except (ValueError, OSError):
        return False


@unittest.skipUnless(english_model_available(), "libtesseract or the system eng model is not installed")
class InspectStepsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.image = root / "hello.png"
        image = Image.new("RGB", (900, 220), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 64)
        except OSError:
            font = ImageFont.load_default(size=64)
        draw.text((40, 30), "HELLO 123", fill="black", font=font)
        draw.text((40, 120), "OCR STEPS", fill=(40, 40, 160), font=font)
        image.save(cls.image, dpi=(300, 300))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_steps(self, name, **kwargs):
        return steps.inspect_steps(self.image, ["eng"], "system", 6, self.root / name, **kwargs)

    def test_steps_match_tesseract_binary_and_cli_text(self):
        result = self.run_steps("otsu", debug_images=False)
        ids = [step["id"] for step in result["steps"]]
        self.assertEqual(ids, ["input", "grey", "threshold", "blobs", "blocks", "paragraphs", "lines", "words", "recognition", "output", "fields"])
        by_id = {step["id"]: step for step in result["steps"]}
        self.assertEqual(by_id["threshold"]["values"]["replica_match_percent"], 100.0)
        self.assertEqual(by_id["threshold"]["values"]["resolution_used"], 300)
        self.assertGreaterEqual(by_id["lines"]["values"]["count"], 2)
        self.assertEqual(len(by_id["grey"]["values"]["histogram"]), 256)
        for step in result["steps"]:
            for name in step["images"].values():
                self.assertTrue((self.root / "otsu" / name).is_file(), name)
        text = by_id["output"]["values"]["text"]
        self.assertIn("HELLO", text)
        line = by_id["recognition"]["lines"][0]
        self.assertTrue(line["images"]["original"] and line["images"]["binary"])
        self.assertEqual(line["words"][0]["language"], "eng")
        self.assertTrue(line["words"][0]["symbols"][0]["choices"])
        # the inspector reads with the same library the dashboard CLI path uses
        cli = TesseractEngine().recognize(self.image, ["eng"], psm=6, variant="system")
        self.assertEqual(re.sub(r"\s+", " ", cli.full_text).strip(), re.sub(r"\s+", " ", text).strip())

    def test_fields_step_reruns_every_nid_row(self):
        step = next(item for item in self.run_steps("fields-step", debug_images=False)["steps"] if item["id"] == "fields")
        self.assertEqual([row["field"] for row in step["rows"]], [])  # no NID labels on a "HELLO 123" image
        self.assertEqual(step["values"]["calls"], 0)
        self.assertIn(step["values"]["reader"], ("capi", "cli"))
        self.assertEqual(set(step["values"]["fields"]), set(step["values"]["page_fields"]))

    def test_fields_step_can_be_switched_off(self):
        ids = [item["id"] for item in self.run_steps("no-fields", debug_images=False, fields=False)["steps"]]
        self.assertNotIn("fields", ids)

    def test_adaptive_and_sauvola_maps_match_tesseract(self):
        for method in ("adaptive_otsu", "sauvola"):
            result = self.run_steps(method, threshold={"method": method}, recognize=False, debug_images=False)
            threshold = next(step for step in result["steps"] if step["id"] == "threshold")
            self.assertEqual(threshold["values"]["replica_match_percent"], 100.0, method)
            self.assertIn("threshold_map", threshold["images"])
            self.assertLessEqual(threshold["values"]["threshold_min"], threshold["values"]["threshold_max"])

    def test_manual_threshold_and_layout_only(self):
        result = self.run_steps("manual", threshold={"method": "manual", "value": 100}, recognize=False, debug_images=False)
        by_id = {step["id"]: step for step in result["steps"]}
        self.assertEqual(by_id["threshold"]["values"]["threshold"], 100)
        self.assertEqual(by_id["threshold"]["values"]["replica_match_percent"], 100.0)
        self.assertNotIn("output", by_id)
        self.assertNotIn("mean_confidence", by_id["recognition"]["values"])
        self.assertIn("layout_ms", result["timings"])

    def test_debug_step_explains_psm_without_page_layout(self):
        result = self.run_steps("debug6", recognize=False, debug_images=True)
        debug = next(step for step in result["steps"] if step["id"] == "debug")
        self.assertEqual(debug["images"], {})
        self.assertIn("PSM 6", debug["explain"])


if __name__ == "__main__":
    unittest.main()
