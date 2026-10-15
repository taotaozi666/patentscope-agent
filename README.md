# Patentscope Agent

基于 Python、Streamlit、LangChain 和 Chroma 的专利知识库问答应用，支持 TXT、Markdown 和 PDF。

## 启动

需要 Python 3.12+ 和 uv：

```powershell
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在 `.env` 中配置 `API_KEY`、`BASE_URL`、`CHAT_MODEL` 和 `EMBEDDING_MODEL`。未配置密钥时可使用离线演示模式。

```powershell
uv run patentscope-agent
```

也可运行 `uv run streamlit run frontend/ui.py` 启动页面。

## 功能

- 后台导入 TXT、Markdown 和 PDF，并将文本向量化后保存到本地 Chroma。
- PDF 支持原生文字、基础表格、区域图片、来源定位和质量检查；可抽取部分中国专利著录项、摘要及权利要求。
- 保存页面诊断、区域处理计划和实际执行记录；解析拒绝时保留原件和诊断快照。图片支持区域裁剪 OCR、位置融合和冲突复核，扫描正文支持逐块来源；复杂表格与版面处理仍在逐步实现。
- 可选使用本机 Tesseract 识别 PDF 图片文字；需安装 Tesseract 及 `chi_sim`、`eng` 语言包。
- 表格默认使用 PyMuPDF，支持框线合并格和文字对齐候选；可选 Docling 后端。逻辑表支持多级表头关联、保守跨页续接、单位与注释来源，并将单元格关联传入切片，见 [表格说明](app/pdf/tables/README.md)。
- 表格按行组/列组切片；检索命中后从同一解析版本的原始资产补齐行标识、表头和单位证据，缺失或冲突时标记不完整，见 [证据补全说明](app/pdf/tables/retrieval.md)。
- 切片支持 `text`、`table`、`image` 类型和来源属性；查询页、回答引用及检索来源可显示关联图片。
- 专利介绍类问题会加强对著录信息、发明名称和摘要片段的召回。回答依据不足时会说明当前检索结果缺少依据。
- 提供知识库问答、文档与切片查询、后台导入和 PDF 解析检查页面。
- 解析检查支持结构表、跨页片段、单元格及行列表头/单位的分色高亮；表格有独立使用状态和人工复核记录，已阻断表格的数值不作为回答证据，见 [质量与复核说明](app/pdf/tables/quality.md)。

## 处理流程

```text
文档 → 解析与切片 → 向量化 → Chroma
问题 → 向量检索 → 相关片段与来源 → 回答及引用
```

PDF 原文、解析结果和切片保存在配置的数据目录；后台任务数据由 SQLite 保存。复杂 PDF 版式和专利抽取效果仍需结合实际文档评估。当前默认使用向量与本地关键词 BM25 的混合召回，通过 RRF 融合排序，尚未加入独立重排模型；配置、恢复和评估边界见 [混合检索说明](app/rag/hybrid.md)。

## PDF 解析回归

混合页面与复杂表格的合成样本、独立预期和当前能力基线见 [回归样本说明](tests/pdf_regression/README.md)。

```powershell
uv run python -m unittest tests.test_pdf_regression -v
uv run python scripts/check_pdf_regression.py
```

检查脚本默认关闭 OCR，将 8 个样本 PDF 和缺口报告保存至 `.documents/pdf-regression/`。报告生成成功不代表复杂表格能力全部通过。

## 后端模块

后台导入与向量写入由独立的 `DocumentImportService` 编排，`RAGService` 保留检索及兼容入口。配置与版本、表格证据读取和紧凑资产存储见 [后端处理链路](app/indexing/README.md)。
