# Patentscope Agent：LangChain + Chroma RAG

知识库聊天应用。当前后台上传页支持 UTF-8 TXT 和 Markdown 多文件导入，文档和向量由 Chroma 本地持久化；聊天记录保存在浏览器会话内存。PDF 等格式可通过扩展加载器支持。

## 快速开始

需要 Python 3.12+ 和 uv，在项目根目录执行：

```powershell
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在已有 `.env` 中配置密钥、模型与服务地址。


```powershell
uv run patentscope-agent
# 或直接启动页面
uv run streamlit run frontend/ui.py
```

## 项目中的五个组件

| 组件 | 本项目实现 | 职责 |
| --- | --- | --- |
| 文档对象 | `langchain_core.documents.Document` | `page_content` 保存正文，`metadata` 保存来源、格式、标题、编号等 |
| 加载器 | `UTF8DocumentLoader(BaseLoader)`、`LoaderRegistry` | 从上传字节读取 TXT/MD，统一输出 Document，无需临时文件 |
| 分割器 | `RecursiveCharacterTextSplitter` + Markdown 标题处理 | 一级标题仅存元数据，二级及以下章节标题与正文一起切分和向量化 |
| 嵌入模型 | `SemanticEmbedder(Embeddings)` | 调用千问兼容向量接口，实现 `embed_documents` 和 `embed_query` |
| 向量数据库 | `langchain_chroma.Chroma` / `ChromaKnowledgeStore` | 存储文档、元数据和向量，使用 Chroma 原生余弦检索 |

嵌入模块使用 LangChain 标准 `Embeddings` 接口，保留现有 `BASE_URL` 配置与批量校验逻辑。当前使用项目的千问适配器，而非 `DashScopeEmbeddings` 的原生 DashScope SDK。后续可通过 `create_embeddings()` 返回其他 LangChain 嵌入模型。


## 页面操作

1. 选择检索模式；有密钥时默认选择中文语义向量。
2. 在“导入任务”页批量上传 `.txt` / `.md`，点击“提交后台解析”。
3. 在任务页查看各文件的状态、阶段、进度、错误和重试结果；解析期间可切换页面。
4. 完成后在聊天页提问，或在“知识库查询”页查看片段、元数据和向量。

知识库查询不调用模型。聊天侧栏的“载入示例文档”仍为同步操作；正常文件导入请使用后台任务页。载入示例后可以问“RAG 是什么？”。后台任务和单文件大小限制可通过 `INGESTION_MAX_FILES`、`INGESTION_MAX_FILE_BYTES`、`INGESTION_MAX_BATCH_BYTES` 配置，任务状态保存在 `INGESTION_DIRECTORY` 指定的本地 SQLite 数据库中。失败任务可重试，排队任务可取消，服务重启后会恢复未完成任务。当前限制为每批最多 20 个文件、单文件 50 MB、整批 200 MB。

导入页会提示本批文件互相重复、内容已存在或同名文件将更新，并允许逐项取消提交。成功提交后会清空已选择的文件。导入相同内容且解析配置未变化时会跳过，并在任务页标记“重复跳过”。同名文件内容变化时会先写入新版本，成功后清理旧片段；失败时保留旧版本可检索。失败任务重试会复用稳定的文档/片段 ID，避免重复写入。去重范围是当前 Chroma 集合；历史上已入库但没有内容哈希的记录会在同名文件下一次成功导入时替换。清空聊天仅删除当前会话聊天；“清空知识库和聊天”删除当前模型集合的全部数据及当前会话聊天。

## 文档导入与分割

```text
上传字节 → 注册加载器 → Document(page_content, metadata)
         → 递归分割后的 Document
         → LangChain Chroma.add_documents
         → Embeddings.embed_documents → Chroma 本地保存
```

默认每片段最多 600 个 Python 字符，目标重叠 100 字符。可在 `.env` 调整 `CHUNK_SIZE`、`CHUNK_OVERLAP`，要求 `0 <= overlap < size`。
向量 API 每次最多发送 10 个片段，并发数由 `EMBEDDING_CONCURRENCY` 控制，默认 2，允许设置为 1 到 8；遇到服务限流时可调低。

- TXT：将整篇正文交给递归分割器，按空行、换行、中文句末标点、空格、字符依次尝试切分。TXT 中的 `#` 保留为普通文本。
- Markdown：先识别 `#` 到 `######` 的标题路径与正文段落，再递归切分；一级标题只保存在 metadata 中，二级及以下章节标题附在每个正文片段前参与向量化。章节标题占用字符预算，过长章节标题会提示导入失败；一级标题不占用正文预算。
- Markdown 的纯标题不会独立入库；围栏代码中的井号不识别为标题。正文段落之间不合并、不重叠，长段内部才尝试重叠。
- 递归分割优先保持分隔单位，实际重叠可能小于目标值，也可能没有重叠；不足一个片段的短正文不会强行补齐。
- metadata 保留 `source`、`format`、`chunk`；Markdown 按实际标题保留 `document_title`（一级标题文本）、`heading_path`（完整标题路径）、`section_path`（二级及以下章节路径）。入库时补充 `document_id` 和 UTC `created_at`。这些元数据默认不参与向量计算，可以在知识库查询页查看。

例如 `## 04．雨伞大学` 和后面的正文保存在同一片段，文档总标题“一座城市把自己的名字忘在冰箱里”只保存在元数据中。没有章节标题时仅向量化正文。规则变化仅作用于新导入的数据；需要更新已有记录时，清空对应集合后重新导入。

千问每批最多处理 10 条文本，验证返回索引并归一化向量。导入不会调用聊天模型。写入失败时尝试删除本次导入的片段，避免保留部分数据。

## 聊天链路

```text
问题 → embed_query → Chroma 余弦检索 → Top K 片段
     → 本轮引用资料 + 最近最多 12 条聊天消息
     → 千问聊天模型 → 完整回答 + 引用来源 → 保存成功的一轮聊天
```

Top K 默认 3，页面可选 1–8。Chroma 返回 cosine 距离，页面相似度显示 `1 - distance`，仅保留大于浮点容差的正相似度结果。目前没有独立重排模型。

短问题（少于 12 字符）或包含“它”“这个”“刚才”的问题，会追加最近两个用户问题重新检索；这仍是规则处理，没有调用模型重写。

提示词集中在 `app/prompt/prompt.py`，要求仅依据当前资料回答并使用 `[1]` 等引用标记。引用编号为本轮结果顺序，与原始片段编号不同。聊天模块继续使用现有千问兼容客户端，关闭思考与流式调用，完成后一次显示回答。失败的一轮不保存历史。

`app/history/history.py` 管理聊天记录。全部历史留在会话内存，发送给模型时截取最近 12 条，不会因此删除更早的历史。

## 离线演示与语义模式

| 模式 | 嵌入 | 检索 |
| --- | --- | --- |
| 离线演示（字符匹配） | 字符及相邻组合词频，稳定哈希到 2048 维并归一化 | Chroma 余弦检索，可能存在哈希碰撞，不具备语义理解 |
| 中文语义向量 | `EMBEDDING_MODEL` 对应的千问文本向量 | Chroma 余弦检索，依赖远程向量服务 |

两种模式都使用 Chroma，没有原有的内存字典存储。离线检索不联网；但配置了密钥且命中资料时，仍调用千问聊天模型。无密钥时展示检索原文；无命中时直接提示资料不足。

## Chroma 持久化与集合

默认目录为项目根目录 `.chroma/`，已加入 `.gitignore`。无需单独启动数据库服务，重启应用后文档和向量仍保留。

集合名称为 `CHROMA_COLLECTION` 前缀加模型身份摘要。语义集合按向量模型及 `BASE_URL` 区分，演示集合使用独立身份；模型变更会连接对应集合，原集合不会被删除，切换回来仍可查询。仅更换密钥不会改变集合。不同维度的模型不应使用同一身份配置。

同一目录和模型集合由各浏览器会话共享，当前没有用户权限或租户隔离。聊天历史仍各自独立。清空只影响当前集合。旧的内存知识库没有自动迁移，需要重新上传。

查询页通过 `KnowledgeStore` 接口访问数据。关键词查询当前扫描符合来源的文档和元数据，仅为当前页加载向量；数据规模增长后可替换查询实现。

## 配置

`app/conf/config.py` 读取 `.env` 与系统环境变量，同名环境变量优先，每次页面操作重新获取配置。

主名称显式定义后优先于别名，即使其为空也不回退到别名。密钥不进入页面控件，Settings 的字符串表示隐藏密钥。切换模型或修改检索配置会重新连接服务并清空当前会话聊天，不删除已有知识库。

## 扩展文件格式

实现 LangChain `BaseLoader`，然后注册即可，页面上传格式自动读取注册表：

```python
from app.documents import loaders
# 工厂接收 source: str、content: bytes，返回 BaseLoader 实例。
loaders.register("pdf", YourPDFLoader)
```

加载器的 `load()` / `lazy_load()` 输出 Document，可在 metadata 中保存页码等信息。当前未安装 PDF、Word 解析依赖，也不接受这些格式。新增格式无需改动聊天与数据库流程。

后台终端保留导入、问题向量、存储检索、检索结果、追问、模型请求、完整回答和耗时日志。`ContextVar` 关联同一次操作的步骤。日志不会主动打印密钥，但包含文档和聊天内容。

## 统一数据模型

`app/schemas/` 集中定义项目内部的数据模型，通过统一入口导入：

```python
from app.schemas import ChunkRecord, StoreStats, QueryResult, Hit, ChatMessage, Settings
```

- `knowledge.py`：片段记录 `ChunkRecord`、统计 `StoreStats`、分页结果 `QueryResult` 和检索结果 `Hit`。
- `chat.py`：`ChatMessage`，使用 TypedDict 描述聊天消息字典结构。
- `configuration.py`：`Settings`，使用 dataclass 保存配置快照，密钥仍不出现在 repr 中。

这些是用于层间传递数据的结构，不是 Chroma 内部表或 ORM 模型；dataclass 和 TypedDict 本身也不提供运行时字段校验。`schemas` 仅依赖标准库和同包模型，不依赖业务模块。配置读取、历史处理、存储操作仍由对应业务模块负责；`app/models.py` 继续封装千问模型调用。`KnowledgeStore` 和 `HistorySession` 是行为接口 Protocol，保留在对应业务模块。

## 验证

```powershell
uv run python -m unittest discover -s tests -v
```

测试包括配置、加载注册、Markdown 标题、递归切分、真实 Chroma 写入/查询/检索/重连/集合隔离/清空、模拟千问请求、历史、日志与启动入口。测试使用独立临时集合与数据库，不调用真实云端模型。

## 最终目标：可评估的专利 PDF RAG 管道

当前先完成 TXT / Markdown 上传和后台解析；PDF 专利解析、结构抽取、混合检索和评估属于后续目标，尚未实现。目标验收标准如下：

- 使用至少 20 份版式不同的专利 PDF，抽取摘要、说明书、权利要求及引用关系；元数据和权利要求结构抽取正确率达到 95%。
- 使用结构化分块与混合检索，并建立人工标注问题集；至少 30 个测试问题的 Recall@5 达到 85%。
- 回答逐条附页码和段落引用；检索不到充分证据时明确拒答。
- 为扫描件、跨页段落、表格、OCR 错误各保留失败样例，便于回归测试。
- 提供可重复运行的评估命令和结果报告，记录数据集版本、指标及运行配置。

开发评估集时，将人工标注的字段真值、有效证据片段和失败类型纳入版本控制或受控数据目录，并在报告中分别展示抽取各字段的准确率，避免单一总分掩盖某种版式的退化。详细目标设计见整体设计文档。

## 目标设计文档

以下描述最终规划，不代表当前已实现功能：

- [整体设计文档](docs/整体设计文档.md)：目标能力、模块边界、数据模型、导入与问答链路、版本一致性、权限、部署和实施顺序。
- [模块思维导图](docs/模块思维导图.md)：模块职责的 Mermaid 版本，可在支持 Mermaid 的 Markdown 查看器中阅读。
- [可展开的思维导图](docs/模块思维导图.html)：浏览器打开本地文件即可查看，无需联网。
