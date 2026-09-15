import hashlib
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nid_ocr_lab.engines import tessdata


class TessdataVariantTests(unittest.TestCase):
    def setUp(self):
        tessdata._HASH_CACHE.clear()
        tessdata._VERSION_CACHE.clear()

    def test_languages_are_recursive_sorted_and_exclude_osd(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "script").mkdir()
            for name in ("eng.traineddata", "ben.traineddata", "osd.traineddata", "script/Bengali.traineddata"):
                (root / name).write_bytes(b"model")

            variant = tessdata.TessdataVariant("best", "tessdata_best", root)

            self.assertEqual(variant.languages(), ["ben", "eng", "script/Bengali"])
            self.assertEqual(variant.model_path("script/Bengali"), root / "script/Bengali.traineddata")

    def test_missing_variant_directory_has_no_languages(self):
        variant = tessdata.TessdataVariant("missing", "missing", Path("/does/not/exist"))
        self.assertEqual(variant.languages(), [])

    def test_system_directory_prefers_environment_then_binary_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            configured = root / "configured"
            configured.mkdir()
            binary = root / "bin" / "tesseract"
            binary.parent.mkdir()
            binary.touch()
            discovered = root / "share" / "tessdata"
            discovered.mkdir(parents=True)

            with patch.dict(os.environ, {"TESSDATA_PREFIX": str(configured)}):
                self.assertEqual(tessdata.system_tessdata_dir(), configured)
            with patch.dict(os.environ, {}, clear=True), patch.object(tessdata.shutil, "which", return_value=str(binary)):
                self.assertEqual(tessdata.system_tessdata_dir(), discovered)

    def test_system_directory_uses_documented_fallback(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(tessdata.shutil, "which", return_value=None):
            self.assertEqual(tessdata.system_tessdata_dir(), Path("/opt/homebrew/share/tessdata"))

    def test_variant_discovery_is_stable_and_ignores_custom_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            system = root / "system"
            best = root / "best"
            custom = root / "custom"
            for directory in (system, best, custom / "zeta", custom / "alpha"):
                directory.mkdir(parents=True)
            (custom / "README.txt").write_text("not a model directory", encoding="utf-8")

            with (
                patch.object(tessdata, "BEST_DIR", best),
                patch.object(tessdata, "CUSTOM_DIR", custom),
                patch.object(tessdata, "system_tessdata_dir", return_value=system),
            ):
                variants = tessdata.list_variants()

            self.assertEqual([variant.id for variant in variants], ["system", "best", "custom/alpha", "custom/zeta"])
            self.assertEqual(variants[2].label, "custom: alpha")

    def test_get_variant_defaults_to_system_and_rejects_unknown_ids(self):
        system = tessdata.TessdataVariant("system", "system", Path("/models/system"))
        with patch.object(tessdata, "list_variants", return_value=[system]):
            self.assertIs(tessdata.get_variant(None), system)
            with self.assertRaisesRegex(ValueError, "Unknown Tesseract model variant: best"):
                tessdata.get_variant("best")

    def test_sha256_handles_missing_files_and_caches_unchanged_models(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "ben.traineddata"
            model.write_bytes(b"trained model")
            expected = hashlib.sha256(b"trained model").hexdigest()

            with patch.object(tessdata.hashlib, "sha256", wraps=hashlib.sha256) as digest:
                self.assertEqual(tessdata.file_sha256(model), expected)
                self.assertEqual(tessdata.file_sha256(model), expected)
            self.assertEqual(digest.call_count, 1)
            self.assertIsNone(tessdata.file_sha256(Path(tmp) / "missing.traineddata"))

    def test_model_version_is_parsed_and_cached(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "ben.traineddata"
            model.write_bytes(b"trained model")
            completed = subprocess.CompletedProcess([], 0, "Version: 4.00.00alpha:ben:synth20170629\n", "")

            with patch.object(tessdata.shutil, "which", return_value="/usr/bin/combine_tessdata"), patch.object(
                tessdata.subprocess, "run", return_value=completed
            ) as run:
                self.assertEqual(tessdata.model_version(model), "4.00.00alpha:ben:synth20170629")
                self.assertEqual(tessdata.model_version(model), "4.00.00alpha:ben:synth20170629")

            run.assert_called_once_with(
                ["combine_tessdata", "-d", str(model.resolve())], capture_output=True, text=True, timeout=20
            )

    def test_model_version_tolerates_missing_tool_and_command_timeout(self):
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "eng.traineddata"
            model.write_bytes(b"model")
            with patch.object(tessdata.shutil, "which", return_value=None):
                self.assertIsNone(tessdata.model_version(model))
            with patch.object(tessdata.shutil, "which", return_value="combine_tessdata"), patch.object(
                tessdata.subprocess, "run", side_effect=subprocess.TimeoutExpired("combine_tessdata", 20)
            ):
                self.assertIsNone(tessdata.model_version(model))

    def test_describe_models_includes_missing_language_entries(self):
        variant = tessdata.TessdataVariant("best", "best", Path("/models/best"))
        with patch.object(tessdata, "file_sha256", side_effect=["abc", None]), patch.object(
            tessdata, "model_version", side_effect=["v1", None]
        ):
            with patch.object(Path, "is_file", side_effect=[True, False]), patch.object(
                Path, "stat", return_value=type("Stat", (), {"st_size": 123})()
            ):
                models = tessdata.describe_models(variant, ["ben", "missing"])

        self.assertEqual(
            models,
            [
                {
                    "language": "ben",
                    "path": "/models/best/ben.traineddata",
                    "bytes": 123,
                    "sha256": "abc",
                    "version": "v1",
                },
                {
                    "language": "missing",
                    "path": "/models/best/missing.traineddata",
                    "bytes": None,
                    "sha256": None,
                    "version": None,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
