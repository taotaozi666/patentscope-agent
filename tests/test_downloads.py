import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory
from pathlib import Path
from streamlit.testing.v1 import AppTest
from app.downloads import is_download_request, resolve_downloads
from app.history import ChatHistory
from app.schemas import Hit, Settings


class DownloadsTests(unittest.TestCase):
    def test_intent(self):
        for text in ['下载跨模态互检索的方法、装置、服务器及存储介质', '请帮我下载 CN116049459A', '把这个专利的原文件给我', '下载这个']:
            self.assertTrue(is_download_request(text), text)
        for text in ['这份专利如何实现文件下载？', '下载失败是什么原因？', '不要下载', '你是谁']:
            self.assertFalse(is_download_request(text), text)

    def test_resolution_and_context(self):
        doc = SimpleNamespace(id='one', title='测试专利', source='CN116049459A.pdf')
        store = Mock()
        store.documents.return_value = [doc]
        store.query.return_value = SimpleNamespace(items=[])
        search = Mock(return_value=[Hit(doc.source, '', .9, 1, {'document_id': 'one'})] * 3)
        history = ChatHistory()
        self.assertEqual(resolve_downloads('下载测试专利', store, history, search)[0], ['one'])
        self.assertEqual(resolve_downloads('下载CN116049459A', store, history, search)[0], ['one'])
        self.assertEqual(resolve_downloads('下载CN999999999A', store, history, search)[0], [])
        search.assert_not_called()
        self.assertEqual(resolve_downloads('下载相关技术文档', store, history, search)[0], ['one'])
        history.add_turn('介绍', '结论 [1]', search.return_value)
        self.assertEqual(resolve_downloads('下载这个', store, history, search)[0], ['one'])
        history.add_turn('下载这个', '已找到', retrieval=False, downloads=['one'])
        self.assertEqual(history.messages()[-1]['downloads'], ['one'])
        self.assertEqual(history.recent_questions(), ['介绍'])

    def test_download_ui_survives_rerun_without_model(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            settings = Settings('', 'chat', 'embedding', chroma_directory=directory,
                                pdf_directory=str(Path(directory) / 'files'))
            with patch('app.config.get_settings', return_value=settings), patch('frontend.source_components.get_settings', return_value=settings):
                page = AppTest.from_file(str(Path('frontend/ui.py').resolve())).run()
                rag = page.session_state['rag']
                rag.add_document('example.txt', '用于下载测试的原文件内容')
                with patch.object(rag, 'search') as search, patch('app.models.create_client') as client:
                    page.chat_input[0].set_value('下载example.txt').run()
                    self.assertFalse(page.exception)
                    self.assertEqual(len(page.get('download_button')), 1)
                    search.assert_not_called()
                    client.assert_not_called()
                    page.run()
                    self.assertFalse(page.exception)
                    self.assertEqual(len(page.get('download_button')), 1)
                    page.chat_input[0].set_value('下载这个').run()
                    self.assertFalse(page.exception)
                    self.assertEqual(len(page.get('download_button')), 2)
                for file in (Path(directory) / 'files' / 'originals').iterdir():
                    file.unlink()
                page.run()
                self.assertFalse(page.exception)
                self.assertTrue(any('重新上传原文件' in item.value for item in page.caption))

if __name__ == '__main__':
    unittest.main()
