import unittest
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from nid_ocr_lab.dashboard import server
from nid_ocr_lab.engines import tesseract as adapter
from nid_ocr_lab.engines.tessdata import TessdataVariant

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def row(level, block, par, line, word, left, top, width, height, conf, text):
    return f"{level}\t1\t{block}\t{par}\t{line}\t{word}\t{left}\t{top}\t{width}\t{height}\t{conf}\t{text}"


SAMPLE_TSV = "\n".join([
    HEADER,
    row(1, 0, 0, 0, 0, 0, 0, 800, 500, -1, ""),
    row(4, 1, 1, 1, 0, 10, 10, 300, 30, -1, ""),
    row(5, 1, 1, 1, 1, 10, 10, 60, 30, 91.5, "নাম:"),
    row(5, 1, 1, 1, 2, 80, 12, 230, 28, 80.5, "ডনিয়েল"),
    row(5, 1, 1, 2, 1, 10, 50, 70, 25, -1, "Name:"),
    row(5, 1, 1, 2, 2, 90, 50, 150, 26, 70, "DONIEL"),
    row(5, 2, 1, 1, 1, 10, 200, 80, 20, 95, "ID"),
    row(5, 2, 1, 1, 2, 5, 200, 20, 20, 50, "   "),
])


class FakeVariant(TessdataVariant):
    def model_path(self, language):
        return Path(__file__)  # any existing file


class TesseractTests(unittest.TestCase):
    def test_words_are_grouped_into_lines(self):
        blocks, words, text = adapter.parse_tsv_lines(SAMPLE_TSV)
        self.assertEqual([block.text for block in blocks], ["নাম: ডনিয়েল", "Name: DONIEL", "ID"])
        self.assertEqual(text, "নাম: ডনিয়েল\nName: DONIEL\n\nID")
        first = blocks[0].bounding_box
        self.assertEqual((first.x, first.y, first.width, first.height), (10, 10, 300, 30))
        self.assertAlmostEqual(blocks[0].confidence, 0.86)
        # conf -1 is ignored rather than averaged in
        self.assertAlmostEqual(blocks[1].confidence, 0.70)
        self.assertEqual(len(words), 5)

    def test_command_sets_model_folder_oem_psm_and_tsv_renderer(self):
        command = adapter.TesseractEngine().build_command(
            Path("card.png"), "ben+eng", Path("/models/best"), psm=6, oem=1, dpi=450
        )
        self.assertEqual(command[:3], ["tesseract", "card.png", "stdout"])
        joined = " ".join(command)
        for fragment in ("--tessdata-dir /models/best", "-l ben+eng", "--oem 1", "--psm 6", "--dpi 450", "tessedit_create_tsv=1"):
            self.assertIn(fragment, joined)
        # the "tsv" config file only exists in the system tessdata folder
        self.assertNotEqual(command[-1], "tsv")
        self.assertNotIn("--dpi", adapter.TesseractEngine().build_command(Path("c.png"), "eng", Path("/m"), 6, 1, None))

    def test_parse_osd(self):
        output = "Page number: 0\nOrientation in degrees: 90\nRotate: 270\nOrientation confidence: 5.12\nScript: Latin\nScript confidence: 2.00\n"
        orientation = adapter.parse_osd(output)
        self.assertEqual(orientation.rotate, 270)
        self.assertAlmostEqual(orientation.confidence, 5.12)
        self.assertEqual(orientation.script, "Latin")
        self.assertIsNone(adapter.parse_osd("Too few characters. Skipping this page"))

    def test_estimated_dpi_and_metadata(self):
        self.assertEqual(adapter.estimated_dpi("Estimating resolution as 569\n"), 569)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            Image.new("RGB", (20, 20), "white").save(path, dpi=(300, 300))
            self.assertEqual(adapter.resolve_dpi(path, None), (300, "image metadata"))
            Image.new("RGB", (20, 20), "white").save(path)
            self.assertEqual(adapter.resolve_dpi(path, None), (None, "tesseract estimate"))


class DashboardTesseractFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "card.png"
        Image.new("RGB", (60, 40), "white").save(self.image)
        variant = FakeVariant("system", "system", Path(self.tmp.name))
        for target in ("get_variant", "describe_models"):
            replacement = (lambda _id: variant) if target == "get_variant" else (lambda *_: [])
            patcher = patch.object(adapter, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        which = patch.object(adapter.shutil, "which", return_value="/usr/bin/tesseract")
        which.start()
        self.addCleanup(which.stop)

    def fake_run(self, command, **kwargs):
        if "0" in command and command[command.index("--psm") + 1] == "0":
            return CompletedProcess(command, 1, "", "Too few characters")
        return CompletedProcess(command, 0, SAMPLE_TSV, "")

    def test_manual_rotation_is_one_call_and_sweep_keeps_candidates_separate(self):
        with patch.object(adapter.subprocess, "run", side_effect=self.fake_run) as run:
            single = server.run_tesseract(self.image, "ben+eng", "as_uploaded", "0", "system", 6, "single")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(single["calls"], 1)
            self.assertEqual(single["input_path"], self.image)

            run.reset_mock()
            sweep = server.run_tesseract(self.image, "ben+eng", "as_uploaded", "90", "system", 6, "sweep")
        self.assertEqual(run.call_count, len(server.SWEEP_PSMS))
        self.assertEqual([candidate["psm"] for candidate in sweep["candidates"]], server.SWEEP_PSMS)
        self.assertTrue(all(candidate["score_kind"] == "heuristic, not accuracy" for candidate in sweep["candidates"]))
        # the chosen result is one PSM's text, never a union of all of them
        self.assertEqual(sweep["ocr"].full_text.count("DONIEL"), 1)
        self.assertTrue(sweep["input_path"].name.endswith("-rot90.png"))

    def test_auto_rotation_falls_back_to_four_way_check_when_osd_fails(self):
        with patch.object(adapter.subprocess, "run", side_effect=self.fake_run) as run:
            result = server.run_tesseract(self.image, "eng", "as_uploaded", "auto", "system", 6, "single")
        self.assertEqual(result["orientation"]["method"], "rotation-check")
        # OSD + four orientation candidates; the winning candidate's OCR is reused
        self.assertEqual(run.call_count, 5)
        self.assertEqual(result["calls"], 5)


if __name__ == "__main__":
    unittest.main()
