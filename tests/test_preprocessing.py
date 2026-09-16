import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

from nid_ocr_lab.dashboard import server
from nid_ocr_lab.dashboard.filters import nid_ink
from nid_ocr_lab.preprocessing.nid_pipeline import inspect_preprocessing


class NIDPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "nid.png"
        image = Image.new("RGB", (420, 260), (224, 216, 170))
        draw = ImageDraw.Draw(image)
        draw.ellipse((120, 30, 300, 210), fill=(204, 140, 48))
        draw.rectangle((300, 48, 390, 150), fill=(120, 92, 80))
        draw.text((32, 70), "Name", fill=(20, 20, 20))
        draw.text((32, 115), "ID NO 1234567890", fill=(140, 20, 20))
        image.save(self.image, dpi=(300, 300))

    def tearDown(self):
        self.tmp.cleanup()

    def test_nid_ink_v1_matches_existing_filter(self):
        result = inspect_preprocessing(self.image, self.root / "v1", "nid_ink_v1")
        with Image.open(self.image) as source, Image.open(result["final_image"]["path"]) as final:
            diff = ImageChops.difference(nid_ink(source.convert("RGB")), final.convert("RGB"))
        self.assertFalse(diff.getbbox())
        self.assertEqual([step["id"] for step in result["steps"]], ["input", "green_channel", "autocontrast", "final"])

    def test_nid_ink_v2_writes_visible_steps_and_final_image(self):
        result = inspect_preprocessing(self.image, self.root / "v2", "nid_ink_v2")
        ids = [step["id"] for step in result["steps"]]
        self.assertEqual(ids[0], "input")
        self.assertIn("background_estimate", ids)
        self.assertIn("ink_mask", ids)
        self.assertEqual(ids[-1], "final")
        self.assertTrue(Path(result["final_image"]["path"]).is_file())
        for step in result["steps"]:
            for name in step["images"].values():
                self.assertTrue((self.root / "v2" / name).is_file(), name)


    def test_nid_ink_v2_controlled_contrast_uses_contrast_as_final(self):
        result = inspect_preprocessing(self.image, self.root / "v2_contrast", "nid_ink_v2_contrast")
        controlled = self.root / "v2_contrast" / "04-controlled-contrast.png"
        self.assertTrue(controlled.is_file())
        with Image.open(controlled) as expected, Image.open(result["final_image"]["path"]) as final:
            diff = ImageChops.difference(expected.convert("RGB"), final.convert("RGB"))
        self.assertFalse(diff.getbbox())
        self.assertEqual(result["recipe"]["label"], "NID ink v2 controlled contrast")

    def test_preprocess_route_pattern_rejects_traversal(self):
        self.assertIsNotNone(server.PREPROCESS_FILE_PATTERN.match("/api/preprocess/20260915-120000-0123abcd/99-final-ocr.png"))
        for path in (
            "/api/preprocess/20260915-120000-0123abcd/../x.png",
            "/api/preprocess/../../etc/passwd",
            "/api/preprocess/20260915-120000-0123abcd/a.txt",
        ):
            self.assertIsNone(server.PREPROCESS_FILE_PATTERN.match(path))

    def test_dashboard_payload_uses_same_input_preparation(self):
        with patch.object(server, "PREPROCESS_ROOT", self.root / "preprocess"):
            result = server.run_preprocess_payload(
                {
                    "recipe": "off",
                    "input": "as_is",
                    "mode": "original",
                    "image_path": str(self.image),
                }
            )
        self.assertEqual(result["recipe"]["id"], "off")
        self.assertEqual(result["image"]["input_mode"], "as_uploaded")
        self.assertEqual(result["image"]["filter_mode"], "original")
        self.assertTrue(Path(result["final_image"]["path"]).is_file())
        self.assertGreaterEqual(result["timings"]["preprocess_ms"], 0)


if __name__ == "__main__":
    unittest.main()
