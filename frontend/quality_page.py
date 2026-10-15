"""PDF 解析检查页面：自动报告、原文对照和人工复核。"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from frontend.image_components import render_image

from app.pdf.preview import render_source_page, render_source_region
from app.quality.service import inspect_saved_pdf
from app.schemas import Settings
from app.schemas.quality import ReviewDecision
from app.storage.pdf_repository import PdfRepository
from app.storage.quality_repository import QualityRepository
from app.storage.index_status import pdf_index_status
from frontend.quality_layout import sync_quality_panel_height
from frontend.table_review import render_table_review

logger = logging.getLogger(__name__)
STATUS = {"passed": "通过", "review": "待复核", "blocked": "阻断"}
DECISIONS: dict[str, ReviewDecision] = {"待复核": "pending", "通过": "passed", "有问题": "problem"}


def render_quality_page(settings: Settings) -> None:
    st.caption("规则检查用于发现风险，不代表人工标注准确率。人工结论单独保存，不覆盖自动报告；附图内容尚未理解。")
    repository = PdfRepository(settings.pdf_directory)
    quality = QualityRepository(settings.pdf_directory)
    runs = repository.list_runs()
    if not runs:
        st.info("暂无 PDF 解析记录。请先在导入任务页上传原生 PDF。")
        return
    by_id = {run["id"]: run for run in runs}
    run_id = st.selectbox("选择解析文档", list(by_id), format_func=lambda value:
                         f"{by_id[value]['source']} · {by_id[value]['content_hash'][:8]} · {value[:8]}", key="quality_document")
    if run_id is None:
        return
    run = repository.get_run(run_id)
    if run is None:
        st.warning("解析记录已不存在。")
        return
    report = quality.latest(run_id)
    if st.button("运行 / 更新自动检查", key=f"quality_check_{run_id}"):
        try:
            report = inspect_saved_pdf(settings.pdf_directory, run_id)
        except Exception:
            logger.exception("解析检查失败")
            st.error("检查未完成，请查看服务端日志。")
            return
    if report is None:
        st.info("这份文档还没有检查报告，点击上方按钮生成。")
        return

    checked_at = datetime.fromisoformat(report.checked_at).astimezone(ZoneInfo("Asia/Shanghai"))
    st.write(f"**自动结论：{STATUS[report.status]}** · 规则 {report.rule_version} · 检查时间 {checked_at:%Y-%m-%d %H:%M:%S}（北京时间）")
    st.caption(f"解析：已保存 · 解析器：{run['parser_version']} · 历史成功入库：{'是' if by_id[run_id]['status'] == 'indexed' else '否'}")
    try:
        indexes = pdf_index_status(settings, run_id, report.chunk_ids)
        st.caption("当前配置的 Chroma：" + "；".join(f"{name}：{value}" for name, value in indexes.items()))
    except Exception:
        logger.exception("读取索引状态失败")
        st.warning("当前 Chroma 状态读取失败，不能确定是否完整入库。解析检查仍可继续。")
    cols = st.columns(4)
    cols[0].metric("页数", len(report.pages))
    cols[1].metric("索引片段", len(report.chunk_ids))
    cols[2].metric("待复核项", sum(item.status == "review" for item in report.issues))
    cols[3].metric("阻断项", sum(item.status == "blocked" for item in report.issues))
    if report.status == "passed":
        st.success("本轮规则没有发现异常，仍需核对文字、标题、阅读顺序和权利要求。")
    elif report.status == "blocked":
        st.error("检测到结构或来源映射异常，新导入会在向量化之前停止；已有向量不会因此删除，复核标记不会自动修复数据。")
    else:
        st.warning("存在需要人工核对的项目。待复核提示不会阻止向量入库。")

    st.dataframe([asdict(page) for page in report.pages], column_config={
        "page_no": "物理页码", "text_chars": "提取字符数（不含页眉页脚）", "paragraphs": "段落数",
        "tables": "表格数", "images": "图片区域数", "chunks": "片段数",
    }, hide_index=True, width="stretch")
    reviews = {item.target: item for item in quality.reviews(report.id)}
    document_review = reviews.get("document")
    review_names = {"pending": "待复核", "passed": "通过", "problem": "有问题"}
    st.caption(f"人工文档结论：{review_names[document_review.decision] if document_review else '未复核'} · 已记录 {len(reviews)} 个复核范围；不作为问答启用门槛。")

    patent = run.get("patent")
    patent_targets: dict[str, tuple[str, list[dict]]] = {}
    if patent:
        with st.expander("专利结构：名称、章节和权利要求", expanded=True):
            for name, label in (("title", "专利名称"), ("publication_number", "公开号")):
                field = patent.get(name)
                st.write(f"**{label}**：{field['value'] if field else '未识别'}")
                if field:
                    patent_targets[name] = (label, field["spans"])
            st.write(f"**抽取器**：{patent['extractor_version']} · {len(patent['claims'])} 项权利要求")
            st.caption("当前为中文原生专利规则，尚未评估准确率。")
            for section in patent["sections"]:
                pages = sorted({span["page_no"] for span in section["spans"]})
                st.write(f"{section['title']}：物理页 {', '.join(map(str, pages))}")
                patent_targets[f"section:{section['kind']}"] = (section["title"], section["spans"])
            st.dataframe([{"编号": claim["number"], "引用权利要求": ", ".join(map(str, claim["depends_on"])) or "未检测到引用",
                           "物理页": ", ".join(map(str, sorted({span["page_no"] for span in claim["spans"]})))}
                          for claim in patent["claims"]], hide_index=True, width="stretch")
            for claim in patent["claims"]:
                patent_targets[f"claim:{claim['number']}"] = (f"权利要求 {claim['number']}", claim["spans"])
            with st.expander("完整专利结构及来源"):
                st.json(patent)
    else:
        st.caption("当前解析记录没有专利结构。普通文档不启用专利规则；旧版专利记录需重新上传，更新检查不会重新解析文件。")
    patent_target = st.selectbox("定位专利字段 / 章节 / 权利要求", ["", *patent_targets],
                                 format_func=lambda value: patent_targets[value][0] if value else "不选择专利结构",
                                 key=f"quality_patent_{run_id}") if patent_targets else None
    patent_spans = patent_targets[patent_target][1] if patent_target else []
    issue_by_id = {item.id: item for item in report.issues}
    issue_id = st.selectbox("定位检查问题", ["", *issue_by_id], format_func=lambda value:
                           "按页浏览（不选择问题）" if not value else
                           f"{STATUS[issue_by_id[value].status]} · 第 {issue_by_id[value].page_no or '—'} 页 · {issue_by_id[value].message}",
                           key=f"quality_issue_{report.id}")
    selected_issue = issue_by_id.get(issue_id or "")
    if selected_issue:
        st.write(selected_issue.message)
        st.caption(f"元素：{selected_issue.element_id or '—'} · 片段：{selected_issue.chunk_id or '—'}")
    table_selection = render_table_review(run, report, run_id, reviews.values())
    default_page = selected_issue.page_no if selected_issue and selected_issue.page_no else 1
    if patent_spans:
        default_page = patent_spans[0]["page_no"]
        st.caption("已定位专利结构；清空结构选择后可按检查问题定位。")
    if table_selection:
        default_page = table_selection["page_no"]
        st.caption("已定位表格；清空逻辑表选择后可按专利结构或检查问题定位。")
    default_page = max(1, min(default_page, run["page_count"]))
    table_key = f"_{table_selection['table_id']}_{table_selection['fragment_id']}_{table_selection['cell_id']}" if table_selection else ""
    page_no = st.selectbox("对照物理页码", list(range(1, run["page_count"] + 1)), index=default_page - 1,
                           key=f"quality_page_{report.id}_{patent_target or issue_id or 'all'}{table_key}")
    if page_no is None:
        return
    elements = [item for item in run["elements"] if item["span"]["page_no"] == page_no]
    chunk_ids = set(report.chunk_ids)
    chunks = [row for row in repository.list_chunks(run_id) if row["metadata"]["pdf_chunk_id"] in chunk_ids
              and row["metadata"]["page_start"] <= page_no <= row["metadata"]["page_end"]]
    highlight = []
    if selected_issue and selected_issue.element_id:
        highlight = [item["span"] for item in elements if item["id"] == selected_issue.element_id]
    if selected_issue and selected_issue.chunk_id:
        try:
            highlight = [span for row in chunks if row["metadata"]["pdf_chunk_id"] == selected_issue.chunk_id
                         for span in json.loads(row["metadata"]["pdf_source_spans"])]
        except (ValueError, TypeError):
            st.warning("问题片段的来源数据损坏，本页将展示原 PDF，不绘制高亮框。")
    if selected_issue and selected_issue.source_spans:
        highlight = selected_issue.source_spans
    if patent_spans:
        highlight = patent_spans
    if table_selection:
        highlight = table_selection["spans"]
    printed_labels = list(dict.fromkeys(item["payload"]["printed_page_label"] for item in elements if item.get("payload", {}).get("printed_page_label")))
    st.caption(f"物理第 {page_no} 页" + (" · 版面页码：" + "；".join(printed_labels) if printed_labels else ""))
    left, right = st.columns(2)
    original: bytes | None = None
    with left:
        st.write("**原 PDF**")
        try:
            original = repository.read_original(run["content_hash"])
            caption = f"物理第 {page_no} 页；彩框对应所选单元格及支持证据。" if table_selection and table_selection["cell_id"] else f"物理第 {page_no} 页；红框为所选问题的来源区域。"
            render_image(render_source_page(original, page_no, highlight), caption=caption)
        except (FileNotFoundError, ValueError, KeyError, TypeError, IndexError):
            st.warning("原文件缺失或来源区域无效，请重新导入。")
    with right:
        st.write("**提取结构与文字**")
        with st.container(height=650, border=True, key="quality_text_scroll"):
            for item in elements:
                with st.expander(f"{item['kind']} · {item['id']} · {item['section_path'] or '未确定章节'}",
                                 expanded=item["kind"] in {"paragraph", "heading"}):
                    payload = item.get("payload", {})
                    origin = payload.get("text_origin", "native")
                    origin_label = {"native": "原生文本", "mixed": "原生文字与 OCR", "ocr": "OCR 识别", "none": "无文字层"}.get(origin, origin)
                    ocr_status = payload.get("ocr_status", "not_applicable")
                    ocr_label = {"recognized": "已识别", "no_text": "未识别到文字", "disabled": "未启用",
                                 "unavailable": "不可用", "not_needed": "无需 OCR", "not_applicable": "不适用"}.get(ocr_status, ocr_status)
                    st.caption(f"类型：{payload.get('doc_type_kwd', 'text')} · 布局：{payload.get('layout_type', item['kind'])} · 文本来源：{origin_label} · OCR：{ocr_label}")
                    if (item["kind"] in {"image", "table"} or payload.get("original_kind") in {"image", "table"}) and original is not None:
                        try:
                            render_image(render_source_region(original, item["span"]), caption=f"第 {page_no} 页 · {item['kind']} 区域裁图")
                        except (ValueError, IndexError, KeyError, TypeError):
                            st.caption("区域裁图暂不可用，可查看左侧原页高亮。")
                    element_text = item["text"] or payload.get("ocr_text", "")
                    st.text(element_text or ("图片区域：未提取到可显示文字" if item["kind"] == "image" else "（无文本）"))
                    if payload.get("ocr_text") and payload["ocr_text"] != element_text:
                        st.text_area("补充 OCR 文字", payload["ocr_text"], height=100, key=f"ocr_{run_id}_{item['id']}")
                    if payload.get("text_conflicts"):
                        st.warning("原生文字与 OCR 存在冲突，备选结果如下，需对照原页确认。")
                        st.json(payload["text_conflicts"])
                    if payload.get("properties"):
                        with st.expander("元素属性"):
                            st.json(payload["properties"])
                    label = f" · PDF 页码标签：{item['span']['page_label']}" if item['span']['page_label'] else ""
                    st.caption(f"物理第 {item['span']['page_no']} 页{label} · 区域：{item['span']['bbox']}")
                    if item["kind"] == "table":
                        st.json(item["payload"])
            with st.expander("本页索引片段及来源映射"):
                for row in chunks:
                    st.caption(row["metadata"]["pdf_chunk_id"])
                    st.text(row["text"])
                    if row["metadata"].get("parent_id"):
                        parent = repository.get_object(row["metadata"]["parent_id"])
                        with st.expander("完整父对象与独立检索上下文"):
                            st.text(row.get("context_text", row["metadata"].get("context_text", "")))
                            if parent:
                                st.text(parent["raw_content"])
                                st.json(parent)
                    try:
                        st.json(json.loads(row["metadata"]["pdf_serialized_map"]))
                    except (ValueError, TypeError):
                        st.warning("字符来源映射格式损坏。")
            if patent and patent_target and patent_target.startswith("claim:"):
                claim = next(item for item in patent["claims"] if f"claim:{item['number']}" == patent_target)
                with st.expander("所选权利要求完整正文（可跨页）", expanded=True):
                    st.text(claim["text"])

    sync_quality_panel_height()
    st.write("**人工复核**")
    targets = {"document": "整份文档", f"page:{page_no}": f"第 {page_no} 页"}
    if selected_issue:
        targets[f"issue:{selected_issue.id}"] = "当前检查问题"
    if table_selection and table_selection["reviewable"]:
        targets[f"table:{table_selection['table_id']}"] = "当前逻辑表（含所有物理片段）"
        if table_selection["cell_reviewable"]:
            targets[f"cell:{table_selection['table_id']}:{table_selection['cell_id']}"] = "当前单元格"
    target = st.selectbox("复核范围", list(targets), format_func=lambda value: targets[value], key=f"quality_target_{report.id}")
    if target is not None:
        previous = reviews.get(target)
        labels = list(DECISIONS)
        initial = next((label for label, value in DECISIONS.items() if previous and value == previous.decision), "待复核")
        with st.form(f"quality_review_{report.id}_{target}"):
            decision = st.radio("人工结论", labels, index=labels.index(initial))
            note = st.text_area("复核说明", value=previous.note if previous else "", help="例如：第 4 页段落错序，权利要求 2 被拆开。")
            if st.form_submit_button("保存复核"):
                quality.review(report.id, target, DECISIONS[decision], note)
                st.success("复核已保存。")
                st.rerun()
    with st.expander("已保存的人工复核"):
        st.dataframe([asdict(review) for review in reviews.values()], hide_index=True)
    exported = {"document": {"source": run["source"], "content_hash": run["content_hash"], "title": run["title"]},
                "patent": patent, "table_fragments": run.get("table_fragments", []), "logical_tables": run.get("logical_tables", []),
                "report": asdict(report), "reviews": [asdict(review) for review in reviews.values()]}
    st.download_button("下载检查报告 JSON", json.dumps(exported, ensure_ascii=False, indent=2),
                       file_name=f"{run['source']}.quality.json", mime="application/json")
