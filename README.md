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
- 可选使用本机 Tesseract 识别 PDF 图片文字；需安装 Tesseract 及 `chi_sim`、`eng` 语言包。
- 切片支持 `text`、`table`、`image` 类型和来源属性；查询页、回答引用及检索来源可显示关联图片。
- 专利介绍类问题会加强对著录信息、发明名称和摘要片段的召回。回答依据不足时会说明当前检索结果缺少依据。
- 提供知识库问答、文档与切片查询、后台导入和 PDF 解析检查页面。

## 处理流程

```text
文档 → 解析与切片 → 向量化 → Chroma
问题 → 向量检索 → 相关片段与来源 → 回答及引用
```

PDF 原文、解析结果和切片保存在配置的数据目录；后台任务数据由 SQLite 保存。复杂 PDF 版式和专利抽取效果仍需结合实际文档评估。当前检索以向量召回为主，尚未加入混合检索与独立重排。
