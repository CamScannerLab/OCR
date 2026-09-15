import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from nid_ocr_lab.dashboard.filters import render_image, rotate_clockwise
from nid_ocr_lab.dashboard.server import preview_rotation, rotated_copy, save_as_uploaded


class PreviewRotationTests(unittest.TestCase):
    def test_preview_matches_ocr_input_rotation(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "card.png"
            image = Image.new("RGB", (40, 20), "white")
            image.putpixel((0, 0), (255, 0, 0))  # red marker at top-left
            image.save(path)
            for degrees in (0, 90, 180, 270):
                payload, info = render_image(str(path), rotation=degrees)
                preview = Image.open(BytesIO(payload))
                with Image.open(rotated_copy(path, degrees)) as ocr_input:
                    self.assertEqual(preview.size, ocr_input.size)
                    self.assertEqual((info.width, info.height), ocr_input.size)
        # clockwise: the top-left marker moves to the top-right corner
        rotated = rotate_clockwise(image, 90)
        self.assertEqual(rotated.size, (20, 40))
        self.assertEqual(rotated.getpixel((19, 0)), (255, 0, 0))

    def test_as_uploaded_filter_keeps_size_and_dpi(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.png"
            image = Image.new("RGB", (2400, 1500), (200, 190, 180))
            image.paste((20, 20, 20), (100, 100, 900, 160))
            image.save(path, dpi=(300, 300))
            original = Path(tmp) / "original.png"
            filtered = Path(tmp) / "filtered.png"
            save_as_uploaded(str(path), original)
            info = save_as_uploaded(str(path), filtered, mode="threshold")
            self.assertEqual((info.width, info.height), (2400, 1500))
            with Image.open(original) as untouched, Image.open(filtered) as output:
                self.assertEqual(untouched.tobytes(), image.tobytes())
                self.assertEqual(output.size, (2400, 1500))
                self.assertEqual(round(output.info["dpi"][0]), 300)
                self.assertNotEqual(output.tobytes(), image.tobytes())
            with self.assertRaises(ValueError):
                save_as_uploaded(str(path), filtered, mode="bogus")

    def test_rotate_query_is_validated(self):
        self.assertEqual(preview_rotation({}), 0)
        self.assertEqual(preview_rotation({"rotate": ["270"]}), 270)
        self.assertIsNone(preview_rotation({"rotate": ["45"]}))


if __name__ == "__main__":
    unittest.main()
