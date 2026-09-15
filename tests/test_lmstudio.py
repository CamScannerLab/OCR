import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nid_ocr_lab.engines import lmstudio_vision as lm


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def reply(content, reasoning="", finish="stop"):
    return {
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 899 if reasoning else 0}},
    }


class LMStudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "card.png"
        self.image.write_bytes(b"\x89PNG\r\n\x1a\n")
        env = patch.object(lm, "load_dotenv", return_value={})
        env.start()
        self.addCleanup(env.stop)

    def test_request_disables_reasoning_by_default(self):
        sent = {}

        def fake_urlopen(request, timeout):
            sent.update(json.loads(request.data))
            return FakeResponse(reply('{"nid_number": "8263067046"}'))

        with patch.object(lm.urllib.request, "urlopen", side_effect=fake_urlopen):
            fields, ocr = lm.LMStudioVisionEngine().recognize_fields(self.image, selected_model="qwen/qwen3.5-9b")
        self.assertEqual(sent["reasoning_effort"], "none")
        self.assertEqual(sent["max_tokens"], 900)
        self.assertEqual(fields["nid_number"], "8263067046")
        self.assertEqual(ocr.metadata["finish_reason"], "stop")

    def test_reasoning_effort_can_be_omitted(self):
        with patch.dict(lm.os.environ, {"LM_STUDIO_REASONING_EFFORT": "default", "LM_STUDIO_MAX_TOKENS": "2048"}):
            self.assertIsNone(lm.reasoning_effort())
            self.assertEqual(lm.max_tokens(), 2048)

    def test_empty_answer_after_reasoning_explains_the_cause(self):
        with self.assertRaises(RuntimeError) as caught:
            lm._extract_content(reply("", reasoning="thinking...", finish="length"))
        self.assertIn("899 tokens reasoning", str(caught.exception))
        self.assertIn("LM_STUDIO_REASONING_EFFORT", str(caught.exception))

    def test_embedding_models_are_not_listed(self):
        payload = {"data": [{"id": "qwen2.5-vl-7b-instruct"}, {"id": "text-embedding-nomic-embed-text-v1.5"}]}
        with patch.object(lm.urllib.request, "urlopen", return_value=FakeResponse(payload)):
            self.assertEqual(lm.list_models(), ["qwen2.5-vl-7b-instruct"])


class DefaultModelTests(unittest.TestCase):
    def test_qwen25_vl_is_preferred_when_loaded(self):
        self.assertEqual(lm.preferred_model(["qwen/qwen3.5-9b", "qwen2.5-vl-7b-instruct"]), "qwen2.5-vl-7b-instruct")
        self.assertEqual(lm.preferred_model(["qwen/qwen3.5-9b"]), "qwen/qwen3.5-9b")
        self.assertEqual(lm.preferred_model([]), "")


if __name__ == "__main__":
    unittest.main()
