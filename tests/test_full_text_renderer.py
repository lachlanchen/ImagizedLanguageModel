import os
import sys
import unittest

import numpy as np

from ilm.visual_lm.full_text_renderer import CompleteLineRenderer


class FullTextRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.version_info[:2]!=(3,12) or "libglib-2.0.so" not in os.environ.get("LD_PRELOAD",""):
            raise unittest.SkipTest("run these real renderer checks in the native Python 3.12 environment")
        cls.renderer=CompleteLineRenderer()

    def test_complete_bilingual_text_and_spaces(self):
        text="Hello, world. 中国文字：言。"
        image,metadata=self.renderer.render(text)
        self.assertEqual(metadata["source_characters"],len(text))
        self.assertEqual(image.shape[0],32);self.assertEqual(image.shape[1]%64,0)
        self.assertTrue(set(np.unique(image))=={0,255})
        self.assertTrue(np.all(image[[0,-1]]==255))
        spaced=self.renderer.measure("English words")
        joined=self.renderer.measure("Englishwords")
        self.assertGreater(spaced["logical_width_px"],joined["logical_width_px"])
        self.assertTrue(metadata["complete_source"])

    def test_exact_repeat_and_explicit_source_rejection(self):
        first,_=self.renderer.render("Language and 汉字")
        second,_=self.renderer.render("Language and 汉字")
        self.assertTrue(np.array_equal(first,second))
        for text in (" ","line one\nline two","unknown \U0010ffff"):
            with self.assertRaises(ValueError):self.renderer.render(text)
        with self.assertRaises(OverflowError):self.renderer.render("中"*1000,maximum_blocks=4)


if __name__=="__main__":unittest.main()
