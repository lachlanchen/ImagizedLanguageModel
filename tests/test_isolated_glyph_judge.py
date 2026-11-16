"""Physical fidelity, conservative parsing and ordered reward contracts, CPU only."""
import base64
from io import BytesIO
import json
import unittest

import numpy as np
from PIL import Image

from ilm.training_only import isolated_glyph_judge as judge


class IsolatedGlyphJudgeTest(unittest.TestCase):
    def response(self, value, **extra):
        return dict(model=judge.MODEL, done=True, done_reason='stop',
                    message=dict(content=json.dumps(value, ensure_ascii=False)), **extra)

    def test_pixel_roundtrip_has_no_rendered_replacement(self):
        ink = np.zeros((32, 32), dtype=np.uint8)
        ink[3:28, 7] = 1; ink[17, 2:29] = 1; ink[31, 31] = 1
        packed = judge.packed_cell(ink)
        image = np.array(Image.open(BytesIO(judge.cell_png(packed))))
        self.assertEqual(image.shape, (288, 288, 3))
        self.assertTrue((image[:16] == 255).all())
        restored = (255 - image[16:272:8, 16:272:8, 0]) // 255
        np.testing.assert_array_equal(restored, ink)
        for invalid in (ink.astype(float), ink[:31], ink + 1):
            with self.assertRaises(ValueError): judge.packed_cell(invalid)

    def test_request_contains_only_fixed_prompt_and_one_image(self):
        request = judge.payload(b'physical image bytes')
        self.assertEqual(request['messages'], [dict(role='user', content=judge.PROMPT,
            images=[base64.b64encode(b'physical image bytes').decode()])])
        self.assertFalse(request['think']); self.assertFalse(request['stream'])
        self.assertEqual(request['options']['num_ctx'], 2048)
        self.assertEqual(request['options']['num_predict'], 96)
        self.assertEqual(request['keep_alive'], '60s')

    def test_strict_reading(self):
        self.assertEqual(judge.parse_response(self.response(dict(text='言', uncertain=False)))['text'], '言')
        for value in (dict(text='中中', uncertain=False), dict(text='中', uncertain=True),
                      dict(text='中', uncertain=0), dict(text=None, uncertain=False),
                      dict(text=' ', uncertain=False), dict(text='\x00', uncertain=False),
                      dict(text='中', uncertain=False, correction=True), [], None):
            self.assertFalse(judge.parse_response(self.response(value))['accepted'], value)

    def test_foreign_truncated_duplicate_and_malformed_rejected(self):
        valid = self.response(dict(text='中', uncertain=False))
        for key, value in [('model', 'other'), ('done', False), ('done_reason', 'length')]:
            changed = dict(valid); changed[key] = value
            self.assertFalse(judge.parse_response(changed)['accepted'])
        for content in ('{"text":"中","text":"言","uncertain":false}', 'not JSON', '{}'):
            changed = dict(valid, message=dict(content=content))
            self.assertFalse(judge.parse_response(changed)['accepted'])
        for value in ([], None, {}, dict(valid, message=None)):
            self.assertFalse(judge.parse_response(value)['accepted'])

    def test_order_duplicates_blanks_and_caps(self):
        readings = {x: dict(text=x, accepted=True) for x in '中文 '}
        def score(keys, stopped=True):
            return judge.score_sequence(keys, readings, '中文', [False, False], stopped=stopped)
        self.assertEqual(score('中文')['reward'], 1)
        for wrong in ('文中', '中中文', '中', '中文文'):
            self.assertFalse(score(wrong)['complete_content'])
            self.assertLess(score(wrong)['reward'], 1)
        self.assertTrue(score('中 文')['complete_content'])
        self.assertLess(score('中 文')['reward'], 1)
        self.assertEqual(score(['missing'], stopped=False)['reward'], 0)
        self.assertEqual(score(['missing'])['reward'], 0)
        self.assertEqual(score('  ')['reward'], 0)
        readings['bad'] = dict(text=None, accepted=False)
        self.assertEqual(score(['中', 'bad'])['reward'], 0)
        with self.assertRaises(ValueError): score('中文', stopped=1)


if __name__ == '__main__':
    unittest.main()
