# Q8 权重复核 · q8-external-review-v1

**适用范围：** 新启动的 Level 3 工程探索。控制器配置和 Worker 租约的 `controls.weight_audit` 必须都显式写 `q8-external-review-v1`。Graph 回放、部署身份、同登记噪声校准、采样与正确性、请求隔离、容器内存限制与 OOM 保护照旧。不回填旧回执；历史控制和正式评估仍走其自身协议。

**允许：** CUDA Graph、KV Cache、投机缓冲区、只在计算时使用的 tile 反量化，对原始 Q8_0 codes 和 scales 做可逐项还原的无损排布。**禁止：** 跨算子/请求留存独立的已反量化权重（即使数值恰好可表示）或重新量化的不同权重；包括整层和部分持久镜像、BF16/FP16、FP8/Q4。判定看运行时用途、表示和寿命，不看多占了多少内存或补丁中是否出现可疑单词。

## 运行中：自动取证，不等签字

新的历史最高合格分数、每个已保存的检查点（含正常段边界）、最终选择或连续模式下每次阶段提名，都把内容寻址的 JSON 放到可信宿主的 `run/weight-audit/packets/`。审计绑定 engine/build/quality、E0 对照、完整的 E0 相对源码变更、构建和映射身份、配对计时及对应的静态风险行号；同一 Engine 的静态扫描只做一次。WIP 标注在检查点，不给 WIP 挂上已验证 Engine 的成绩。此流程不访问留出集，也不向 Agent 暴露审计写入权限。

- 记录就绪和请求后的服务显存以及两者的最大读数，不声称有连续分配追踪。只在同节点启动、同 Observer、同权重、同采样工作负载及阶段下计算差值。超过约 300 MiB 仅是复核提示，不扣分，不拒绝。
- W1/W2 的 6.7 ms/token 是项目提出的**未校准筛查估值**，不是硬件实测下界。不凭它认定读了多少权重字节。记录每一对的 TPOT 和投机诊断；没有服务器遥测是 `unknown`，不能当成“未开启投机”。
- 原 GGUF 未更换、数值通过、无静态命中、显存变化小，都不能单独证明没有替代表示。在源码扫描不可能覆盖的路径（特别是 Native、后端库或临时分配）保留不确定性。

## 宿主复核及发布

用 `python -m evolveengine.v3_level3_weight_audit pending <run/weight-audit>` 列出待审。未复核的轨迹、历史最佳、阶段提名**始终是暂定开发观测**。该命令不调模型，也不自动签字。可由 Codex 或研究负责人在真实探索期间异步处理队列，无需暂停 Agent。

1. 查包内 `chosen`、构建产物、E0 相对源码和加载路径。复核 GGUF、实际计算路径、权重表示和存活区间，包括 Python 缓存、模块属性、注册 buffer、Native 分配与持久工作区。静态高风险项和 TPOT/显存异常需要有解释，但不是违规证据。
2. 对**同一个构建**取得独立宿主运行时证据（例如带分配归属与跨请求寿命的外部追踪）；不能让候选自报 dtype 作为唯一证据。整理一份可归档的 JSON：`schema="q8-runtime-witness-v1"`，`chosen` 原样复制，`method` 描述取证方式，`observations` 至少覆盖 `weights_loaded` 与 `post_requests`；每项写 `stage, representation, lifetime, raw_observation`。注明覆盖范围、无法归属的分配及原始证据位置/哈希。报告/证据应脱敏；最终导出包含这份见证。可信操作者对证据来源负责，Runner 只校验格式、哈希与绑定，不能替代独立判断。
3. 复核报告 JSON 必须含 `packet_sha256, chosen, reviewer, decision, checks, runtime_evidence_sha256`。`checks` 分别写 `gguf_integrity, runtime_representation, persistent_alternate_copies` 的具体结论。运行 `python -m evolveengine.v3_level3_weight_audit record <run/weight-audit> <packet-sha256> <report.json> <runtime-evidence>`。`approved` 需要上述运行时见证；覆盖不足写 `inconclusive`，可追加证据；`rejected` 是同 Engine/Build 的否决，不靠换一个证据包洗掉。只在可信宿主执行，无 Agent RPC。
4. 非连续模式提名后状态是 `pending_weight_review`。连续模式 `finish_exploration` 仍然只是阶段提名，Agent 可继续；要正常收尾可在宿主创建 `run/pause-after-segment`，等当前段及作业结算。控制器不为等审计而继续消耗模型预算。
5. 在可信宿主用 `python -m evolveengine.v3_level3_delivery <exploration> <pinned-E0-archive> <new-output-dir>` 发布；从已暂停的连续运行导出时额外传 `--proposal-segment N`，必须是 Agent 明确提名且冷构建验证过的段。未批准对应的**最终提名包**、构建身份不符、证据改动、未结清、已有拒绝或计时/正确性不满足发布条件，均不能导出。历史开发回执不改写；只有成功导出的包标记 `externally_reviewed`。

即使复核通过，也不保证检测所有 Native 层隐蔽权重副本；签收的是列出的已检查范围。
