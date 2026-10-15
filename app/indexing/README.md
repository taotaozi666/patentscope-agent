# 后端处理链路

`IngestionManager → DocumentImportService → LoaderRegistry → prepare_pdf → parse_pdf → build_chunks → enhance_context → check_pdf → PdfRepository / QualityRepository → embeddings → KnowledgeStore`

`RAGService` 负责检索与回答入口，旧的 `add_file / add_document / add_documents` 委托独立的 `DocumentImportService`。后台任务直接使用导入服务。离线模式仍在导入服务初始化时移除模型调用配置。

## 配置与身份

- `app/pdf/profile.py` 集中创建表格选项、解析参数、上下文配置、配置分组和索引指纹。
- 原件由内容哈希定位；解析运行由原件、来源名称、解析器版本定位；切片由解析运行、切片版本、预算、父对象及原文定位。
- 模型上下文配置包含模型和服务端点，即使增强失败回退，也有独立切片身份。密钥轮换不改变资产身份。
- 本次解析器为 native-v15、语义对象 v8、切片 v13。旧资产仍可读取；重新导入采用新版本，成功写入后沿用既有旧索引清理流程。

## 表格原始依据和派生视图

原始单元格保存在解析记录的 `table_fragments`。新的表格语义对象使用 `evidence_schema_version=2`，通过逻辑表和片段 ID 引用原始单元格，不再内嵌完整 `evidence_cells`。

`app/pdf/tables/evidence.py` 提供统一索引、归属校验和同片段支持单元格读取。检索仍先验证命中切片、来源、解析版本和父对象身份；旧 schema v1 从父对象内嵌证据读取。跨页重复表头继续保留在物理片段中。

父对象仅保存表头和单位引用；切片需要的支持文字和坐标在构建切片时投影。为兼容现有消费者，切片元数据中的支持证据和元素网格暂时保留。它们不会替代检索时从原始资产读取的证据。

## 存储与向量

- 原件统一由 `SourceRepository` 原子保存，保存 PDF 前核对内容哈希。
- `pdf_runs` 新记录不再重复内嵌 `objects`；对象独立保存在 `pdf_objects`，`get_run` 恢复既有完整返回格式。历史内嵌记录继续可读。
- 资产仓库共用 30 秒 SQLite 连接超时；按运行查询切片和对象有索引。
- 导入服务在写入前检查向量批次数量、最终数量、维度和有限数值。任何错误都不会调用索引写入；向量库自身保留原有分批失败回滚和完整文档去重行为。

人工复核界面与判定策略未调整。混合召回已在后续接入，见 [混合检索说明](../rag/hybrid.md)；未增加独立重排或自动重新解析历史文档。

## 验证

新增 `tests/test_backend_refactor.py` 覆盖端点与密钥的身份变化、模型失败回退的切片隔离、紧凑及历史资产恢复、错配原件拒绝、异常向量阻断和表格片段归属。

完整 Docling 推理与真实扫描准确率仍需要相应运行依赖、模型和业务样本验收。
