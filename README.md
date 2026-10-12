# Patentscope Agent

面向专利资料的知识库问答项目，使用 Python、Streamlit、LangChain 和 Chroma。当前支持 TXT / Markdown，后续扩展为可评估的专利 PDF RAG 管道。

## 如何启动

需要 Python 3.12+ 和 uv。在项目根目录执行：

```powershell
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，配置 `API_KEY`、`BASE_URL`、`CHAT_MODEL` 和 `EMBEDDING_MODEL`。无密钥时可使用离线演示模式。

```powershell
uv run patentscope-agent
```

也可以直接启动页面：

```powershell
uv run streamlit run frontend/ui.py
```

进入“导入任务”上传文档，完成后在“知识库问答”提问，或在“知识库查询”查看文档和片段。

## 整体设计

```text
文档 → 后台导入 → 加载与切分 → 向量化 → Chroma 知识库
问题 → 向量检索 → 相关片段 → 模型根据资料回答并标注来源
```

- `frontend/`：知识库问答、文档查询和导入任务三个页面。
- `app/rag/`：文档加载、切分、向量化与问答流程。
- `app/` 其他模块：模型调用、存储、任务调度、提示词、配置、历史和日志。

文档与向量由 Chroma 本地持久化，后台任务由 SQLite 保存，聊天记录保存在当前会话。应用聚焦知识库问答：问候和助手介绍固定回答，其他问题默认检索；资料不足时要求说明缺少依据，不提供开放式闲聊。

## 最终目标

构建可评估的专利 PDF RAG 管道，目标验收标准：

- 支持至少 20 份不同版式的专利 PDF，抽取摘要、说明书、权利要求及引用关系，元数据和权利要求结构抽取正确率达到 95%。
- 支持结构化分块与混合检索，至少 30 个标注问题的 Recall@5 达到 85%。
- 回答附页码和段落引用，证据不足时明确拒答。
- 覆盖扫描件、跨页段落、表格和 OCR 错误的回归样例。
- 提供可重复运行的评估命令和报告，记录数据集版本、指标及配置。

## 当前进度

已完成：

- UTF-8 TXT / Markdown 多文件后台导入，支持进度、重试、取消和重启恢复。
- 重复内容跳过、同名文档更新，以及 Chroma 本地持久化。
- 两级知识库查询：文档列表与片段详情，支持全文展示、搜索、分页和向量按需读取。
- 语义检索、离线演示、资料引用和追问上下文；已加入资料不足时的回答约束。
- 按模块组织代码，集中管理提示词，支持 INFO / DEBUG 等日志级别。

待完成：专利 PDF / OCR 解析、专利结构抽取、混合检索与重排、标注评估集和指标报告。资料相关性与拒答效果尚需通过实际问题集验证。

运行现有测试：

```powershell
uv run python -m unittest discover -s tests -v
```
