# 表格后端（第 5 项）

默认 `PDF_TABLE_BACKEND=native`：PyMuPDF 先检测框线；没有结果且存在至少三行文字对齐候选时，在候选裁剪区尝试无框线检测。框线格的跨度来自几何边界，无框线单元格来源使用文字内容区域，标记 cell_content，不宣称是精确外框。

可选 `auto` 对复杂原生表、图片页、异常文字层或检测失败尝试 Docling；`docling` 对每页尝试。结果先校验再替换原生结果；失败默认回退，原因保存在页面诊断。`PDF_TABLE_FALLBACK=false` 时失败拒绝导入并保存原件和诊断。

安装可选依赖：`uv sync --extra tables`。在本机预先准备 Docling 模型目录，并设置 `PDF_TABLE_MODE=accurate`（或 fast）、`PDF_TABLE_MODELS_DIRECTORY`。这一步需要完整模型及权重，单有配置文件不足以完成推理；扫描表还需要可用的本机 Tesseract 及语言数据。

Docling 在独立、隐藏的 CPU 子进程内处理去旋转的单页 PDF，限制超时，禁用远程服务并设置模型离线环境；不会在导入时自动联网获取缺失权重。输出统一为原文、跨度、表头标记与原页面坐标，保留引擎原始 JSON；OCR 开启时不猜测每个单元格到底来自哪个文字通道。

验证包含协议模拟和独立安装的官方 Docling Core 数据模型。`scripts/check_docling_contract.py` 可在具备 docling-core 的环境运行。当前工作环境未安装完整 Docling 推理运行时及模型，尚未验证真实神经推理准确率。
