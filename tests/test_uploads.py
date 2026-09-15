import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from nid_ocr_lab.dashboard import uploads


def image_bytes(fmt="PNG", size=(40, 20), exif_orientation=None):
    buffer = BytesIO()
    image = Image.new("RGB", size, "white")
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        image.save(buffer, format=fmt, exif=exif)
    else:
        image.save(buffer, format=fmt)
    return buffer.getvalue()


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(uploads, "upload_root", return_value=Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_png_upload_is_listed_as_sample(self):
        record = uploads.save_upload("../card.png", image_bytes())
        self.assertEqual(record["kind"], "image")
        self.assertEqual(record["filename"], "card.png")
        self.assertTrue(Path(record["page_paths"][0]).is_file())
        samples = uploads.upload_samples()
        self.assertEqual(samples[0]["id"], f"upload:{record['id']}")
        self.assertEqual(list(samples[0]["images"]), ["uploaded image"])
        self.assertIsNone(samples[0]["annotation"])

    def test_exif_rotation_is_applied_to_page(self):
        # orientation 6 = rotate 90 degrees clockwise for display
        record = uploads.save_upload("photo.jpg", image_bytes("JPEG", (40, 20), exif_orientation=6))
        with Image.open(record["page_paths"][0]) as page:
            self.assertEqual(page.size, (20, 40))
        self.assertTrue((Path(record["directory"]) / "original.jpg").is_file())

    def test_pdf_pages_are_rendered(self):
        buffer = BytesIO()
        pages = [Image.new("RGB", (200, 120), "white"), Image.new("RGB", (200, 120), "gray")]
        pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
        record = uploads.save_upload("cards.pdf", buffer.getvalue())
        self.assertEqual(record["kind"], "pdf")
        self.assertEqual(record["pages"], ["page-001.png", "page-002.png"])
        with Image.open(record["page_paths"][0]) as page:
            # 200pt wide at 300 DPI render
            self.assertAlmostEqual(page.width, 200 * 300 / 72, delta=2)
        meta = json.loads((Path(record["directory"]) / "upload.json").read_text())
        self.assertEqual(meta["render_dpi"], 300)
        self.assertEqual(list(uploads.upload_samples()[0]["images"]), ["page 1", "page 2"])

    def test_unsupported_and_empty_uploads_are_rejected_without_leftovers(self):
        with self.assertRaises(ValueError):
            uploads.save_upload("notes.txt", b"hello world")
        with self.assertRaises(ValueError):
            uploads.save_upload("empty.png", b"")
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_delete_is_confined_to_upload_root(self):
        record = uploads.save_upload("card.png", image_bytes())
        for bad in ("..", "../x", "20260101-000000-zzzzzzzz", ""):
            with self.assertRaises(ValueError):
                uploads.delete_upload(bad)
        uploads.delete_upload(record["id"])
        self.assertEqual(uploads.list_uploads(), [])


if __name__ == "__main__":
    unittest.main()
