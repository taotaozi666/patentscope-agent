"""文件获取意图及文档定位；下载地址不参与向量化。"""
import re
from app.patent_identifiers import publication_numbers


def is_download_request(question: str) -> bool:
    text = question.strip()
    if re.search(r"下载失败|无法下载|不能下载|为什么|如何实现|怎么实现|原理|不要下载|不用下载", text):
        return False
    return bool(re.match(r"^(?:请|麻烦|帮我|请帮我|我要|我想|可以|能否|能不能|给我|帮忙|一下|你可以|你能)*\s*(?:下载|获取原文件|获取原文|获取文件)", text)
                or re.search(r"(?:把|将).+(?:给我|发给我|下载)|(?:原文件|原文|PDF|pdf).*(?:给我|下载)", text))


def _key(text):
    return re.sub(r"[\W_]", "", text).lower()


def resolve_downloads(question, store, history, search):
    """返回文档 ID；语义匹配仅提供候选，不宣称精确命中。"""
    documents = store.documents()
    numbers = set(publication_numbers(question))
    target = re.sub(r"^(?:请|麻烦|帮我|请帮我|我要|我想|可以|能否|能不能|给我|帮忙|一下|你可以|你能)*\s*(?:下载|获取原文件|获取原文|获取文件)", "", question).strip()
    if not numbers and (_key(target) in {"", "这个", "这份", "这个专利", "这份专利", "原文件", "原文", "pdf"}
                        or re.search(r"这个|这份|刚才|上述|它", target)):
        for message in reversed(history.messages()):
            if message['role'] != 'assistant':
                continue
            ids = message.get('downloads', [])
            if not ids:
                hits = message.get('hits', [])
                cited = {int(n) for n in re.findall(r"\[([1-9]\d*)\]", message['content'])}
                selected = [h for i, h in enumerate(hits, 1) if i in cited] or hits
                ids = [h.metadata.get('document_id') for h in selected]
            ids = list(dict.fromkeys(i for i in ids if any(d.id == i for d in documents)))
            return ids, '请点击对应文件的下载按钮。' if ids else '请提供要下载的文档标题或公开号。'
        return [], '请提供要下载的文档标题或公开号。'
    exact = []
    for doc in documents:
        if numbers:
            records = store.query(document_id=doc.id, limit=1, include_vectors=False).items
            metadata = records[0].metadata if records else {}
            if numbers.intersection(publication_numbers(doc.source + ' ' + str(metadata.get('publication_number', '')))):
                exact.append(doc.id)
        elif any(_key(name) and _key(name) in _key(target) for name in (doc.title, doc.source)):
            exact.append(doc.id)
    if exact:
        return exact, '已找到以下文档，请选择对应文件下载。'
    if numbers:
        return [], '知识库中未找到该公开号对应的文档，请确认公开号或先上传原文件。'
    hits = search(target or question, 8)
    ids = list(dict.fromkeys(h.metadata.get('document_id') for h in hits
                            if any(d.id == h.metadata.get('document_id') for d in documents)))
    return ids, ('未找到精确匹配，以下为相关候选，请核对文件后下载。' if ids
                 else '未找到相关文档，请提供准确标题、公开号或先上传原文件。')
