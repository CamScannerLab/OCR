import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from nid_ocr_lab.engines import easyocr as adapter
from nid_ocr_lab.engines.tesseract import ocr_result_to_json
from nid_ocr_lab.dashboard.server import run_ocr_payload


class EasyOCRTests(unittest.TestCase):
    def setUp(self):
        version_patch = patch.object(adapter, 'version', return_value='1.7.2')
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def test_language_mapping_and_rejection(self):
        self.assertEqual(adapter.language_codes(['eng', 'ben']), ('bn', 'en'))
        self.assertEqual(adapter.language_codes(['ben']), ('bn',))
        with self.assertRaises(ValueError):
            adapter.language_codes(['fra'])

    def test_numpy_output_serializes_and_preserves_bengali(self):
        reader = Mock()
        reader.readtext.return_value = [(np.array([[1, 2], [20, 2], [20, 12], [1, 12]]), 'নাম Name', np.float32(.9))]
        with patch.object(adapter, 'is_available', return_value=True), patch.object(adapter, 'get_reader', return_value=(reader, True)):
            result = adapter.EasyOCREngine().recognize(Path('card.png'), ['ben', 'eng'], auto_rotate=True)
        self.assertEqual(result.full_text, 'নাম Name')
        self.assertEqual(result.blocks[0].bounding_box.width, 19)
        json.dumps(ocr_result_to_json(result))
        self.assertEqual(reader.readtext.call_args.kwargs['rotation_info'], [90, 180, 270])

    def test_reader_reused_across_engine_instances(self):
        reader = Mock()
        reader.readtext.return_value = []
        with patch.dict(adapter._READERS, {('en',): reader}, clear=True), patch.object(adapter, 'is_available', return_value=True):
            for _ in range(2):
                result = adapter.EasyOCREngine().recognize(Path('card.png'), ['eng'])
                self.assertTrue(result.metadata['reader_cached'])
        self.assertEqual(reader.readtext.call_count, 2)

    def test_dashboard_dispatch_and_empty_result(self):
        reader = Mock()
        reader.readtext.return_value = []
        info = Mock(width=100, height=60, mode='original')
        with patch('nid_ocr_lab.dashboard.server.save_filtered_image', return_value=info), patch.object(adapter, 'is_available', return_value=True), patch.object(adapter, 'get_reader', return_value=(reader, True)):
            result = run_ocr_payload({'engine': 'easyocr', 'image_path': 'card.png', 'language': 'ben', 'rotation': '0'})
        self.assertEqual(result['ocr']['engine'], 'easyocr')
        self.assertEqual(result['ocr']['blocks'], [])
        self.assertIn('request_latency_ms', result)
        self.assertIsNone(reader.readtext.call_args.kwargs['rotation_info'])


if __name__ == '__main__':
    unittest.main()
