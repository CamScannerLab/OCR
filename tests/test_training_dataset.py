import json
import unicodedata
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from nid_ocr_lab.dashboard import server
from nid_ocr_lab.training import dataset, drafts


class DraftLinesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module, name, value in (
            (dataset, "runs_root", self.root / "runs"),
            (dataset, "ground_truth_root", self.root / "gt"),
            (drafts, "ground_truth_root", self.root / "gt"),
            (drafts, "drafts_root", self.root / "drafts"),
        ):
            patcher = patch.object(module, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cards = self.root / "cards"
        self.cards.mkdir()
        for name in ("card_a.png", "card_b.jpg"):
            Image.new("RGB", (300, 120), "white").save(self.cards / name)
        upload = self.root / "uploads" / "20260915-120906-868c9d36"
        upload.mkdir(parents=True)
        (upload / "upload.json").write_text("{}", encoding="utf-8")
        Image.new("RGB", (300, 120), "white").save(upload / "page-001.png")
        Image.new("RGB", (300, 120), "white").save(upload / "original.png")
        self.calls = []

    def fake_ocr(self, payload):
        self.calls.append(payload)
        image = Path(payload["image_path"])
        run_id = dataset.save_run(image, {"blocks": []}, {"sample_id": payload["sample_id"]})
        blocks = [
            {"text": "ডনিয়েল ত্রিপুরা", "confidence": 0.61, "bounding_box": {"x": 60, "y": 10, "width": 150, "height": 20}},
            {"text": "Doniel Tripura", "confidence": 0.9, "bounding_box": {"x": 60, "y": 40, "width": 150, "height": 20}},
            {"text": "মাতা: নিরাজিতা", "confidence": 0.8, "bounding_box": {"x": 10, "y": 70, "width": 200, "height": 20}},
            {"text": "  ", "confidence": None, "bounding_box": {"x": 10, "y": 95, "width": 20, "height": 20}},
        ]
        return {"run_id": run_id, "ocr": {"blocks": blocks}}

    def draft(self, paths, **kwargs):
        with patch.object(server, "run_ocr_payload", side_effect=self.fake_ocr):
            return drafts.draft_lines(paths, "nid_ben", **kwargs)

    def test_drafts_keep_bengali_lines_with_guess_and_review_sheet(self):
        summary = self.draft([self.cards])
        self.assertEqual((summary["images"], summary["lines"], summary["skipped_script"]), (2, 4, 2))
        self.assertEqual(summary["errors"], [])
        payload = self.calls[0]
        self.assertEqual(
            (payload["mode"], payload["strategy"], payload["tesseract_variant"], payload["input"]),
            ("nid_ink", "fields", "best", "as_is"),
        )
        folder = self.root / "drafts" / "nid_ben"
        gts = sorted(folder.glob("*.gt.txt"))
        self.assertEqual(len(gts), 4)
        self.assertEqual(len(list(folder.glob("*.png"))), 4)
        self.assertFalse((self.root / "gt" / "nid_ben").exists())  # nothing reaches training unreviewed
        sidecar = json.loads((folder / gts[0].name.replace(".gt.txt", ".json")).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["status"], "draft")
        self.assertEqual(sidecar["ocr_text"], gts[0].read_text(encoding="utf-8").strip())
        review = (folder / "REVIEW.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(review[0].split("\t"), ["stem", "image", "guess", "confidence", "source_image"])
        self.assertEqual(len(review), 5)
        self.assertIn("0.61", "\n".join(review))

    def test_upload_page_gets_card_id_and_original_is_skipped(self):
        summary = self.draft([self.root / "uploads"], script="all")
        self.assertEqual(summary["images"], 1)
        self.assertEqual(self.calls[0]["sample_id"], "20260915-120906-868c9d36-p001")
        self.assertEqual(summary["lines"], 3)

    def test_failing_image_is_reported_and_batch_continues(self):
        def failing(payload):
            if payload["image_path"].endswith("card_b.jpg"):
                raise RuntimeError("Tesseract failed")
            return self.fake_ocr(payload)

        with patch.object(server, "run_ocr_payload", side_effect=failing):
            summary = drafts.draft_lines([self.cards], "nid_ben")
        self.assertEqual(len(summary["errors"]), 1)
        self.assertIn("card_b.jpg", summary["errors"][0]["image"])
        self.assertEqual(summary["lines"], 2)

    def test_promote_moves_reviewed_lines_and_keeps_empty_ones(self):
        self.draft([self.cards])
        folder = self.root / "drafts" / "nid_ben"
        gts = sorted(folder.glob("*.gt.txt"))
        gts[0].write_text(unicodedata.normalize("NFD", "ডনিয়েল ত্রিপুরা") + " \n", encoding="utf-8")  # unchanged, NFD
        gts[1].write_text("ডনিয়েল ত্রিপুরা সংশোধিত\n", encoding="utf-8")  # corrected
        gts[2].write_text("\n", encoding="utf-8")  # postponed
        existing = self.root / "gt" / "nid_ben"
        existing.mkdir(parents=True)
        (existing / gts[3].name).write_text("already here\n", encoding="utf-8")

        summary = drafts.promote_drafts("nid_ben")
        self.assertEqual(
            (summary["promoted"], summary["edited"], summary["skipped_empty"], summary["skipped_existing"]), (2, 1, 1, 1)
        )
        self.assertIn("training-split", summary["next"])
        moved = existing / gts[1].name
        self.assertEqual(moved.read_text(encoding="utf-8"), "ডনিয়েল ত্রিপুরা সংশোধিত\n")
        sidecar = json.loads(moved.with_name(moved.name.replace(".gt.txt", ".json")).read_text(encoding="utf-8"))
        self.assertEqual((sidecar["status"], sidecar["edited"]), ("verified", True))
        unchanged = existing / gts[0].name
        self.assertEqual(unchanged.read_text(encoding="utf-8"), unicodedata.normalize("NFC", "ডনিয়েল ত্রিপুরা") + "\n")
        self.assertTrue(gts[2].is_file())  # empty draft stays for later
        self.assertTrue(gts[3].is_file())  # clash stays in drafts
        self.assertEqual(len(dataset.line_records("nid_ben")), 2)
        review = (folder / "REVIEW.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(review), 3)  # header + the two drafts still waiting


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for name, value in (("runs_root", root / "runs"), ("ground_truth_root", root / "gt")):
            patcher = patch.object(dataset, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.image = root / "input.png"
        Image.new("RGB", (200, 100), "white").save(self.image)

    def test_padding_is_clamped_to_image(self):
        box = dataset.padded_box({"x": 2, "y": 1, "width": 196, "height": 20}, (200, 100))
        self.assertEqual(box, (0, 0, 200, 24))
        with self.assertRaises(ValueError):
            dataset.padded_box({"x": 300, "y": 300, "width": 10, "height": 10}, (200, 100))

    def test_saved_line_is_nfc_single_line_with_sidecar(self):
        run_id = dataset.save_run(self.image, {"blocks": []}, {"sample_id": "upload:x"})
        decomposed = unicodedata.normalize("NFD", "কী") + "\n  নাম "
        saved = dataset.save_training_lines(
            "nid_ben", run_id, "upload:20260915-card",
            [{"index": 3, "text": decomposed, "ocr_text": "কি নাম", "bounding_box": {"x": 10, "y": 10, "width": 50, "height": 20}}],
        )
        stem = saved[0]["stem"]
        self.assertTrue(stem.startswith("upload_20260915-card__"))
        folder = Path(self.tmp.name) / "gt" / "nid_ben"
        text = (folder / f"{stem}.gt.txt").read_text(encoding="utf-8")
        self.assertEqual(text, unicodedata.normalize("NFC", "কী নাম") + "\n")
        self.assertTrue((folder / f"{stem}.png").is_file())
        sidecar = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["card_id"], "upload:20260915-card")
        self.assertEqual(sidecar["script"], "ben")
        with self.assertRaises(ValueError):
            dataset.save_training_lines("nid_ben", run_id, "c", [{"index": 0, "text": "  ", "bounding_box": {"x": 1, "y": 1, "width": 5, "height": 5}}])
        with self.assertRaises(ValueError):
            dataset.save_training_lines("../escape", run_id, "c", [])

    def test_split_never_puts_one_card_on_both_sides(self):
        records = [{"stem": f"c{card}_{line}", "card_id": f"card{card}"} for card in range(10) for line in range(4)]
        train, evaluation = dataset.split_by_card(records, eval_ratio=0.3, seed=7)
        self.assertEqual(len(train) + len(evaluation), len(records))
        self.assertFalse({r["card_id"] for r in train} & {r["card_id"] for r in evaluation})
        self.assertEqual(len({r["card_id"] for r in evaluation}), 3)
        with self.assertRaises(ValueError):
            dataset.split_by_card([{"stem": "a", "card_id": "only"}])

    def test_runs_are_pruned(self):
        for _ in range(4):
            dataset.save_run(self.image, {}, {})
        dataset.prune_runs(keep=2)
        self.assertEqual(len(list((Path(self.tmp.name) / "runs").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
