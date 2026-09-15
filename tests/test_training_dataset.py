import json
import unicodedata
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from nid_ocr_lab.training import dataset


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
