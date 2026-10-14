import base64
import io
import unittest
from PIL import Image
from streamlit.testing.v1 import AppTest
from app.pdf.parser import _ocr_failure_reason


class ImageDisplayTests(unittest.TestCase):
    def test_thumbnail_keeps_full_resolution_data(self):
        output = io.BytesIO()
        Image.new('RGB', (1600, 900), 'white').save(output, format='PNG')
        content = output.getvalue()
        page = AppTest.from_string("import streamlit as st\nfrom frontend.image_components import render_image\nrender_image(st.session_state.content, width=180)")
        page.session_state['content'] = content
        page.run()
        self.assertFalse(page.exception)
        url = page.image[0].proto.imgs[0].url
        self.assertEqual(base64.b64decode(url.split(',', 1)[1]), content)

    def test_ocr_failure_does_not_always_claim_missing_installation(self):
        self.assertIn('语言数据不可用', _ocr_failure_reason(RuntimeError('No tessdata specified'), 'chi_sim+eng'))
        self.assertIn('OCR 执行失败', _ocr_failure_reason(RuntimeError('render failed'), 'chi_sim+eng'))

if __name__ == '__main__':
    unittest.main()
