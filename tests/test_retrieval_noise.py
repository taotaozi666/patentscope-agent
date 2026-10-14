import unittest
import json
from app.pdf.parser import _remove_overlapping_text
from datetime import datetime
from unittest.mock import Mock
from app.pdf.parser import _mark_repeated_furniture
from app.schemas.pdf import PdfElement, SourceSpan
from app.schemas import ChunkRecord
from app.rag.retrieval import merge_candidates
from app.rag import RAGService
from app.rag.embeddings import DemoEmbedder


def record(index, text, doc='a', **metadata):
    return ChunkRecord(str(index), doc, doc+'.pdf', text, index, [], datetime.now(),
                       {'page_start': index, 'page_end': index, **metadata})


class RetrievalNoiseTests(unittest.TestCase):
    def test_overlaid_text_is_removed_but_separate_numbers_remain(self):
        spans = [{'text': '16', 'bbox': (10, 20, 20, 30)},
                 {'text': '16', 'bbox': (10.01, 20.08, 20.01, 30.08)},
                 {'text': '16', 'bbox': (30, 20, 40, 30)},
                 {'text': '19', 'bbox': (10, 20, 20, 30)}]
        blocks = [{'lines': [{'spans': [span]} for span in spans]}]
        _remove_overlapping_text(blocks)
        self.assertEqual([s['text'] for line in blocks[0]['lines'] for s in line['spans']], ['16', '16', '19'])

    def test_page_number_sequence_filters_margin_only(self):
        elements = [PdfElement(str(i), 'image', str(i),
                    SourceSpan(i, '', (.36,.9198,.56,.936), str(i))) for i in range(1, 5)]
        elements += [PdfElement('body', 'image', '1', SourceSpan(1, '', (.36,.5,.56,.52), 'body')),
                     PdfElement('formula', 'image', '1 + 2', SourceSpan(1, '', (.36,.9198,.56,.936), 'formula'))]
        _mark_repeated_furniture(elements, 4)
        self.assertTrue(all(e.kind == 'furniture' for e in elements[:4]))
        self.assertEqual([e.kind for e in elements[4:]], ['image', 'image'])

    def test_old_index_page_noise_requires_position_and_matching_page(self):
        def metadata(y, kind='image'):
            return {'doc_type_kwd': kind, 'pdf_source_spans': json.dumps([
                {'page_no': 16, 'bbox': [.36,y,.56,y+.02]}])}
        rows = [record(1, '16\n16', **metadata(.9198)),
                record(2, '16', **metadata(.5)), record(3, '19', **metadata(.9198)),
                record(4, '16 + 16', **metadata(.9198)), record(5, '16'),
                record(6, '16', **metadata(.9198, 'table'))]
        hits = merge_candidates([(row, .9) for row in rows])
        self.assertNotIn(1, [location['chunk'] for hit in hits for location in hit.metadata['duplicate_locations']])
        self.assertEqual({location['chunk'] for hit in hits for location in hit.metadata['duplicate_locations']}, {2,3,4,5,6})

    def test_margin_images_and_text_but_not_body_or_full_page(self):
        elements = [PdfElement(str(i), 'image' if i % 2 else 'paragraph', 'CN 116049459 A',
                               SourceSpan(i, '', (0.1, 0.02, 0.8, 0.07), str(i))) for i in range(1, 5)]
        elements += [PdfElement('body', 'paragraph', 'CN 116049459 A', SourceSpan(1, '', (.1,.4,.8,.5), 'body')),
                     PdfElement('scan', 'image', 'CN 116049459 A', SourceSpan(1, '', (0,0,1,1), 'scan'))]
        _mark_repeated_furniture(elements, 4)
        self.assertTrue(all(e.kind == 'furniture' for e in elements[:4]))
        self.assertEqual(elements[4].kind, 'paragraph')
        self.assertEqual(elements[5].kind, 'image')

    def test_merge_preserves_locations_documents_numbers_and_distinct_images(self):
        rows = [record(1, 'CN 116049459 A'), record(2, '参数 12。'), record(3, '参数 12。'),
                record(4, '参数 13。'), record(5, '参数 12。', doc='b'),
                record(6, '图一', image_id='first'), record(7, '图一', image_id='second')]
        hits = merge_candidates([(r, .9) for r in rows])
        self.assertEqual(len(hits), 5)
        self.assertEqual([x['page_start'] for x in hits[0].metadata['duplicate_locations']], [2,3])
        self.assertNotIn('duplicate_locations', rows[1].metadata)

    def test_refills_past_duplicates_without_reembedding(self):
        rows = [record(i, '相同正文') for i in range(1, 10)] + [record(10, '另一段正文')]
        store = Mock()
        store.count.return_value = len(rows)
        store.nearest.side_effect = lambda vector, count: [(r, .9) for r in rows[:count]]
        embedder = DemoEmbedder()
        rag = RAGService(embedder, store=store)
        hits = rag.search('正文', 2)
        self.assertEqual([h.text for h in hits], ['相同正文', '另一段正文'])
        self.assertEqual(store.nearest.call_count, 2)
        self.assertEqual(len(hits[0].metadata['duplicate_locations']), 9)

if __name__ == '__main__':
    unittest.main()
