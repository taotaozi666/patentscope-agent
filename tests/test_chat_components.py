"""全部检索候选在统一的来源折叠区展示。"""

import unittest
from streamlit.testing.v1 import AppTest
from app.schemas import Hit


class ChatComponentsTests(unittest.TestCase):
    def page(self, content, hits):
        page = AppTest.from_string(
            "import streamlit as st\n"
            "from frontend.chat_components import render_message\n"
            "render_message('assistant', st.session_state.content, st.session_state.hits)"
        )
        page.session_state['content'] = content
        page.session_state['hits'] = hits
        return page.run()

    def test_all_candidates_share_one_collapsed_panel(self):
        hits = [Hit('a.pdf', '候选一', .9, 1),
                Hit('b.pdf', '候选二', .8, 35, {'page_start': 3, 'page_end': 3})]
        for answer in ('结论 [2]。', '资料不足。'):
            with self.subTest(answer=answer):
                page = self.page(answer, hits)
                self.assertFalse(page.exception)
                self.assertEqual(len(page.expander), 1)
                panel = page.expander[0]
                self.assertEqual(panel.label, '查看检索来源')
                self.assertFalse(panel.proto.expanded)
                self.assertEqual([item.value for item in panel.text], [hit.text for hit in hits])
                self.assertTrue(any('[2] b.pdf' in item.value and '第 3 页' in item.value
                                    and '相似度 0.800' in item.value for item in panel.markdown))
                self.assertTrue(any('检索候选片段' in item.value for item in panel.caption))

    def test_no_retrieval_has_no_source_panel(self):
        page = self.page('你好！', [])
        self.assertFalse(page.exception)
        self.assertFalse(page.expander)


if __name__ == '__main__':
    unittest.main()
