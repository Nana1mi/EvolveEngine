# P3 上下文 2×2 诊断 — 离线冻结包

冻结版本：`p3-context-2x2-offline-freeze-v1-20261123`。**只完成离线冻结与模拟验收，未授权或执行任何付费调用；未占用 GPU。** 本目录没有 live transport，也不读取 API 凭据，不启动 Pi/r04/W。

复核入口：[冻结清单](frozen/manifest.json) · [完整 SHA-256 列表](SHA256SUMS.txt) · [模拟上游验收回执](offline-rehearsal.json)。

`frozen/manifest.json` 原始字节 SHA-256：`4259814224d61a4f13323b878736ce93483b023c4d33af0256f0103984d648f2`。

## 冻结内容与顺序

- 共用 `system.txt`、`task.txt`、`toy-source.txt` 和 `tools.json`；每个读取长度各自固定 `history-*.json`、`read-*.json`。合成历史中已固定一次 `read` 的工具调用、对应 ID、`reasoning_content` 和其返回字节。短结果是同一文件的前缀，已包括完整目标函数；长条件多出的内容只有注释。实际工具结果短 **8,163 B**、长 **38,532 B**。
- 固定顺序为 `high-short` → `max-long` → `max-short` → `high-long`。每格固定完整 JSON 请求体的原始字节；不任意加 BOM/换行、不重新序列化、不同格不共享模型会话。清单提供每格请求、合成历史和结果的关联及各文件哈希。
- 冻结预留分别为 **2,948 / 9,625 / 2,948 / 9,625**，共 **25,146** 输入 token；按已锁定的公开模板/tokenizer 渲染计数并加登记余量。计划上限 15,000/格、60,000 合计，最多 4 次，每次请求 `max_tokens=4000`，输出合计计划上限 16,000。**这些不是供应商实耗的保证。**
- 固定预期指纹 `bc19ba97cbe016479903222bb1578db6`（冻结清单中哈希固定的先前模型试跑回执的三次调用均观测到它）。这不是实时验证。新诊断任一响应缺失或不匹配就不再发后续请求，不重试或用首格覆盖预期。对已发出的调用，有可信用量则记账；用量缺失或超出预留走 `usage_uncertain` 并停。未发出的预留失败格不记用量。
- 判定只统计有效工具行动、正常结束无有效行动、截断无有效行动；无效或不完整工具结果不会调用工具，遇到 `length` 不评分为成功。有效行动须唯一且完整，目标和参数合法，返回表达式经本地确定性玩具判定，不执行模型提交的代码。只有四格可解释且恰好一格成功才进入另行复验的候选；四格截断则停止参数筛查。`offline-rehearsal.json` 中的“成功格”仅为模拟测试数据，不是模型表现。

## Windows 离线复核

在本目录执行；使用既有、固定哈希的公开 tokenizer 资产，无需凭据或网络。`verify` 会核对外部锚定的 manifest 摘要、原始字节、脚本、依赖和预算。

```powershell
$assets = Join-Path $env:LOCALAPPDATA 'EvolveEngine/deepseek-flash-1017-08c8657f-offline'
$env:PYTHONPATH = Join-Path $assets 'deps' # tokenizers==0.22.1
$sha = '4259814224d61a4f13323b878736ce93483b023c4d33af0256f0103984d648f2'
python -B diagnostic.py verify --assets $assets --expect-sha256 $sha
python -B test_diagnostic.py --assets $assets --expect-sha256 $sha
```

`freeze` 不覆盖已存在的冻结目录。回执在冻结后生成，名称与哈希在列表的“Post-freeze evidence”段，不作为 manifest 的输入。
