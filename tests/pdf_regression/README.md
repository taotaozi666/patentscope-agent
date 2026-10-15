# 混合页面与复杂表格回归样本

队列第 1 项：建立输入、独立预期和当前能力基线。这里不引入新的解析后端。

`fixtures.py` 用 PyMuPDF 生成 8 个合成 PDF，`expected.json` 独立保存目标文字、单元格坐标、行列跨度、表头行、数值支持单元格以及逻辑表分组。生成器不读取预期文件；预期不能从被测解析结果重新生成。

| 样本 | 验证内容 |
| --- | --- |
| native_grid | 原生中文表头、数字、零值与空白格；基础对照 |
| hybrid_page | 同页原生正文与栅格表格，防止漏提取或重复索引 |
| scanned_table | 完全没有文字层的扫描表格 |
| multi_header | 两层表头、跨行和跨列合并 |
| merged_cells | 数据行标识跨两行，数值仍应归属正确实施例 |
| borderless | 无框线、依赖文字对齐的表格 |
| continued_table | 两页续表，保留重复表头来源，合并为一个逻辑表 |
| distinct_tables | 相邻页列数相同但表编号和单位不同，必须保留两张表 |

## 执行

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pdf_regression -v
.\.venv\Scripts\python.exe scripts/check_pdf_regression.py
```

脚本默认不执行 OCR、不请求模型、不读业务知识库。每个样本的 PDF 和报告输出到 `.documents/pdf-regression/`，可直接打开 PDF 核对；`--output` 可更换输出目录。

显式传入 `--ocr` 时请求本机 Tesseract，报告标记为 `tesseract_requested`，实际完成或失败状态见各区域。默认报告标记为 `disabled`。自动化测试中的混合页 OCR 使用固定块模拟，只验证文字融合和重复问题，不代表真实 OCR 准确率。

## 第一轮基线

2026-10-15，解析器 `native-v9:cn-patent-v3`，PyMuPDF 1.28.2，OCR 关闭：

| 样本 | 正确文字且来源位置匹配的单元格 | 当前缺口 |
| --- | --- | --- |
| native_grid | 9/9 | 基础网格检查通过；表头/单位关系和逻辑表模型未提供 |
| hybrid_page | 0/9 | 扫描表区域没有恢复成表格 |
| scanned_table | 0/9 | OCR 关闭时明确拒绝扫描正文页 |
| multi_header | 13/13 | 文字和区域可提取，但缺少显式合并跨度、多级表头关系 |
| merged_cells | 8/8 | 合并格文字和位置可提取，但缺少显式跨行跨度 |
| borderless | 0/9 | 默认框线检测未恢复无框线表 |
| continued_table | 18/18 | 页内网格可提取，跨页逻辑表分组未提供 |
| distinct_tables | 18/18 | 页内网格可提取，逻辑表分组未提供，尚不能验证禁止续接 |

`evaluation.py` 分别衡量表区域、单元格文字、来源位置、行列与跨度、正文出现次数。数值相同不代表正确：先按页码和区域匹配，再比较文字，避免数字互换后仍被误判正确。

`measured_checks_passed` 只表示已测项目通过。页面来源路由、表头/单位关联和逻辑表分组暂时列在 `unavailable_checks`；所有样本的 `target_complete` 都是 false。后续实现这些模型时，需要扩展评估适配器后才能作最终验收。

脚本发现能力缺口时仍完成并保存报告，退出成功仅表示报告生成成功。未预期的运行异常会使脚本失败。

测试另外通过交换数值、错误页码和重复正文验证评估器会拒绝错误输出，防止基线成为只检查“解析没报错”的测试。

这些样本用于结构回归，未覆盖真实专利的低清晰度、倾斜、压缩伪影、复杂字体与全部版式。后续接入解析后端时，需补充经过人工标注的真实样本。

## 第 3 项后的报告变化

当前解析器为 native-v11。报告新增 `page_source_classification` 和 `routing_plan_coverage`，并包含区域诊断及计划/执行/待执行/禁用操作。解析拒绝时也可读取错误携带的诊断。原始的单元格和表结构基线仍保留，不以路由分类通过代替复杂表格解析通过。

实现与阶段边界见 [页面诊断与区域路由](../../app/pdf/inspection.md)。

第 4 项后解析器更新为 native-v12，图片使用区域 OCR，扫描页保留独立正文块，冲突进入质量复核。裁剪变换、融合规则及验证边界见 [区域 OCR 说明](../../app/pdf/ocr.md)。扫描表格目标的行列结构仍未恢复，不能以 OCR 文字出现替代表格验收。

## 第 5、6 项后的基线

native-v14 默认原生路线：native_grid 9/9、multi_header 13/13、merged_cells 8/8、borderless 9/9、continued_table 18/18、distinct_tables 18/18，文字与来源检查通过。多级表头/合并格拓扑已测；新增 header_association 和 logical_table_grouping 比较真实引用与独立标注，明确续表合并、不同表号或单位保持独立。未知表头保留所有数据行。

无框线候选的来源 cell_content 允许落在独立标注单元格内；其区域匹配还必须满足标注行列数。不会把内容框误称为原始完整格线。

默认关闭 OCR，hybrid_page 和 scanned_table 仍有缺口。`--table-backend auto/docling --table-models <本机模型目录>` 可请求可选模型，`--strict-tables` 禁止失败回退。当前只验证了 Docling 适配协议和官方 Core 模型结构，未验证完整模型的真实识别效果。

报告仍将单位规范化和脚注语义列为未完整验收，target_complete 为 false；历史基线保留供比较。当前规则及边界见 [逻辑表说明](../../app/pdf/tables/logical.md)。

第 8 项后 report.json 还保存各样本的 quality_report（表格使用状态、问题、单元格来源）。规则 v6、切片 v12。原生结构检查通过与准确率验收分别记录，扫描/混合样本的缺口仍保留。
