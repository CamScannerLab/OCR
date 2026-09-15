import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nid_ocr_lab.engines.tessdata import TessdataVariant
from nid_ocr_lab.models import OCRResult
from nid_ocr_lab.training import tesstrain_runner as runner


class MakeCommandTests(unittest.TestCase):
    def test_gnu_make_prefers_supported_gmake_and_skips_old_versions(self):
        versions = {
            "/usr/bin/gmake": "GNU Make 4.4.1\n",
            "/usr/bin/make": "GNU Make 3.81\n",
        }

        def which(name):
            return f"/usr/bin/{name}"

        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, versions[command[0]], "")

        with patch.object(runner.shutil, "which", side_effect=which), patch.object(runner.subprocess, "run", side_effect=run):
            self.assertEqual(runner.gnu_make(), "/usr/bin/gmake")

    def test_gnu_make_rejects_missing_or_old_implementations(self):
        completed = subprocess.CompletedProcess([], 0, "GNU Make 3.81\n", "")
        with patch.object(runner.shutil, "which", side_effect=lambda name: "/usr/bin/make" if name == "make" else None), patch.object(
            runner.subprocess, "run", return_value=completed
        ):
            with self.assertRaisesRegex(RuntimeError, "GNU make >= 4.2"):
                runner.gnu_make()

    def test_tesstrain_path_is_relative_to_checkout(self):
        with TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "training" / "tesstrain"
            model_dir = Path(tmp) / "models" / "best"
            with patch.object(runner, "TESSTRAIN_DIR", checkout):
                self.assertEqual(runner.tesstrain_path(model_dir), "../../models/best")

    def test_run_make_records_command_and_reports_nonzero_exit(self):
        with TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "tesstrain"
            checkout.mkdir()
            log = Path(tmp) / "train.log"
            success = subprocess.CompletedProcess([], 0)
            with patch.object(runner, "TESSTRAIN_DIR", checkout), patch.object(
                runner.subprocess, "run", return_value=success
            ) as run:
                runner.run_make("gmake", "lists", {"MODEL_NAME": "nid_ben", "PSM": "7"}, log)

            self.assertIn("$ gmake lists MODEL_NAME=nid_ben PSM=7", log.read_text(encoding="utf-8"))
            self.assertEqual(run.call_args.args[0], ["gmake", "lists", "MODEL_NAME=nid_ben", "PSM=7"])
            self.assertEqual(run.call_args.kwargs["cwd"], checkout)
            self.assertEqual(run.call_args.kwargs["stderr"], subprocess.STDOUT)

            failed = subprocess.CompletedProcess([], 2)
            with patch.object(runner, "TESSTRAIN_DIR", checkout), patch.object(
                runner.subprocess, "run", return_value=failed
            ):
                with self.assertRaisesRegex(RuntimeError, r"`make training` failed \(exit 2\)"):
                    runner.run_make("make", "training", {}, log)


class TrainWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tesstrain = self.root / "tesstrain"
        self.data = self.tesstrain / "data"
        self.langdata = self.data / "langdata"
        self.langdata.mkdir(parents=True)
        (self.tesstrain / "Makefile").touch()
        (self.langdata / "radical-stroke.txt").touch()
        self.python = self.root / "training-venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.touch()
        self.dataset = self.root / "ground-truth" / "nid_ben"
        self.dataset.mkdir(parents=True)
        self.best = self.root / "best"
        self.best.mkdir()
        for language in ("ben", "eng", "osd"):
            (self.best / f"{language}.traineddata").write_bytes(language.encode())
        self.custom = self.root / "custom"
        self.base = TessdataVariant("best", "best", self.best)
        self.split = {"train": ["train_a", "not_generated"], "eval": ["eval_a"], "seed": 7, "eval_cards": ["card-b"]}

    def patches(self):
        return (
            patch.object(runner, "TESSTRAIN_DIR", self.tesstrain),
            patch.object(runner, "TRAINING_VENV_PYTHON", self.python),
            patch.object(runner, "BEST_DIR", self.best),
            patch.object(runner, "CUSTOM_DIR", self.custom),
            patch.object(runner, "dataset_dir", return_value=self.dataset),
            patch.object(runner, "get_variant", return_value=self.base),
            patch.object(runner, "load_split", return_value=self.split),
            patch.object(runner, "gnu_make", return_value="gmake"),
        )

    def test_train_uses_card_split_removes_empty_helpers_and_installs_model(self):
        (self.dataset / "stale.box").touch()
        (self.dataset / "stale.lstmf").touch()
        (self.dataset / "keep.box").write_text("valid", encoding="utf-8")
        calls = []

        def fake_make(make, target, variables, log_path):
            calls.append((make, target, dict(variables), log_path))
            if target == "lists":
                for stem in ("train_a", "eval_a"):
                    (self.dataset / f"{stem}.lstmf").write_bytes(b"compiled")
            elif target == "training":
                (self.data / "nid_custom.traineddata").write_bytes(b"fine tuned")

        patches = self.patches()
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with patch.object(runner, "run_make", side_effect=fake_make):
            info = runner.train(
                "nid_ben",
                "nid_custom",
                start_model="ben",
                max_iterations=125,
                learning_rate=0.0002,
                lang_type="Indic",
                psm=13,
            )

        self.assertEqual([call[1] for call in calls], ["lists", "training"])
        variables = calls[0][2]
        self.assertEqual(
            {key: variables[key] for key in ("MODEL_NAME", "START_MODEL", "PSM", "MAX_ITERATIONS", "LEARNING_RATE")},
            {
                "MODEL_NAME": "nid_custom",
                "START_MODEL": "ben",
                "PSM": "13",
                "MAX_ITERATIONS": "125",
                "LEARNING_RATE": "0.0002",
            },
        )
        self.assertFalse((self.dataset / "stale.box").exists())
        self.assertFalse((self.dataset / "stale.lstmf").exists())
        self.assertTrue((self.dataset / "keep.box").is_file())
        ground_truth_link = self.data / "nid_custom-ground-truth"
        self.assertTrue(ground_truth_link.is_symlink())
        self.assertEqual(ground_truth_link.resolve(), self.dataset.resolve())
        self.assertEqual((self.data / "nid_custom" / "list.train").read_text(), "data/nid_custom-ground-truth/train_a.lstmf\n")
        self.assertEqual((self.data / "nid_custom" / "list.eval").read_text(), "data/nid_custom-ground-truth/eval_a.lstmf\n")

        installed = self.custom / "nid_custom"
        self.assertEqual((installed / "nid_custom.traineddata").read_bytes(), b"fine tuned")
        for companion in ("eng", "ben", "osd"):
            self.assertTrue((installed / f"{companion}.traineddata").is_symlink())
        metadata = json.loads((installed / "model.json").read_text(encoding="utf-8"))
        self.assertEqual((metadata["train_lines"], metadata["eval_lines"]), (1, 1))
        self.assertEqual((info["split_seed"], info["eval_cards"]), (7, ["card-b"]))

    def test_train_rejects_invalid_model_name_before_starting_tools(self):
        with patch.object(runner, "gnu_make") as gnu_make:
            with self.assertRaisesRegex(ValueError, "Model name"):
                runner.train("nid_ben", "../escape")
        gnu_make.assert_not_called()

    def test_train_refuses_to_replace_non_symlink_ground_truth(self):
        conflict = self.data / "nid_custom-ground-truth"
        conflict.mkdir()
        patches = self.patches()
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with self.assertRaisesRegex(RuntimeError, "exists and is not a symlink"):
            runner.train("nid_ben", "nid_custom")

    def test_train_requires_generated_files_on_both_sides_of_split(self):
        patches = self.patches()
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with patch.object(runner, "run_make"):
            with self.assertRaisesRegex(RuntimeError, r"no \.lstmf files for train or eval"):
                runner.train("nid_ben", "nid_custom")


class EvaluateModelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = [
            {"stem": "line-a", "card_id": "card-a", "script": "eng", "text": "HELLO", "image": self.root / "a.png"},
            {"stem": "line-b", "card_id": "card-b", "script": "digits", "text": "123", "image": self.root / "b.png"},
            {"stem": "train-only", "card_id": "card-c", "script": "eng", "text": "SKIP", "image": self.root / "c.png"},
        ]

    def test_evaluate_models_reports_cer_accuracy_latency_scripts_and_errors(self):
        calls = []

        class FakeEngine:
            def recognize(inner_self, image, languages, *, psm, variant):
                calls.append((image.name, languages, psm, variant))
                if image.name == "a.png":
                    return OCRResult(blocks=[], full_text="  HELLO\n", engine="tesseract", latency_ms=10)
                return OCRResult(blocks=[], full_text="124", engine="tesseract", latency_ms=20)

        with (
            patch.object(runner, "load_split", return_value={"eval": ["line-a", "line-b"], "train": ["train-only"]}),
            patch.object(runner, "line_records", return_value=self.records),
            patch.object(runner, "TesseractEngine", return_value=FakeEngine()),
            patch.object(runner, "REPORTS_DIR", self.root / "reports"),
        ):
            report = runner.evaluate_models("nid_ben", ["best:eng+ben"], psm=13)

        model = report["models"][0]
        self.assertEqual(calls, [("a.png", ["eng", "ben"], 13, "best"), ("b.png", ["eng", "ben"], 13, "best")])
        self.assertEqual((report["lines"], report["cards"]), (2, ["card-a", "card-b"]))
        self.assertAlmostEqual(model["cer"], 1 / 8)
        self.assertEqual(model["line_exact_accuracy"], 0.5)
        self.assertEqual(model["mean_latency_ms"], 15)
        self.assertEqual(model["per_script"]["eng"]["line_exact_accuracy"], 1.0)
        self.assertAlmostEqual(model["per_script"]["digits"]["cer"], 1 / 3)
        self.assertEqual(model["errors"], [{"stem": "line-b", "expected": "123", "predicted": "124"}])
        saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertNotIn("report_path", saved)
        self.assertEqual(saved["models"][0]["spec"], "best:eng+ben")

    def test_evaluate_models_rejects_empty_splits_and_malformed_specs(self):
        with patch.object(runner, "load_split", return_value={"eval": []}), patch.object(
            runner, "line_records", return_value=self.records
        ):
            with self.assertRaisesRegex(ValueError, "No eval lines"):
                runner.evaluate_models("nid_ben", ["best:ben"])

        with patch.object(runner, "load_split", return_value={"eval": ["line-a"]}), patch.object(
            runner, "line_records", return_value=self.records
        ):
            with self.assertRaisesRegex(ValueError, "Model spec must be"):
                runner.evaluate_models("nid_ben", ["best"])


if __name__ == "__main__":
    unittest.main()
