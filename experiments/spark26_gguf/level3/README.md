# Level 3 实施登记

开发探索当前的 [判据、实机回执及未放行项](./DEVELOPMENT-GATES.md) 是启动依据。新聊天启动单同时核验两臂校准、同登记六组配对和正负控。旧登记中的 8 MiB 驻留门、逐字重放要求和旧派单顺序不适用于本次探索。

Git 中发布的是代码、配置模板和离线测试，不包含实机回执、启动授权、实验数据或留出集。文中 `evidence/` 链接指向运行方单独保管的材料，不是干净检出所需的测试夹具。

可信控制器的 SSH 地址、登录名和预先核验的 Ed25519 指纹分别由运行方设置 `EE_L3_WORKER_SSH_HOST`、`EE_L3_WORKER_SSH_USER`、`EE_L3_WORKER_SSH_ED25519`（`SHA256:…`）；不设置时连接失败，密码仍通过独立操作员输入。推送代码不会签发启动授权：换用不同字节的控制器或 Worker 时，必须重新核对并绑定相应登记。需要重现字节哈希时使用保留仓库原始换行的副本。

## 当前工程探索的显存与权重契约（2026-11-25）

新启动单使用 [`q8-external-review-v1`](./WEIGHT-AUDIT.md)。实际 GB10 容量、候选容器限额、正确性和请求隔离保留；相对 E0 的显存不再拒绝候选，也不计分。独立 Worker 仍须提供完整的显存、文件和部署证据。历史回执（包括 5,597 MiB 的旧 `violation`）按原协议解释，正式 W 和留出集不随此变更。新增协议需要对应的 Worker 登记、租约以及启动准入证据，本次改动不构成已通过实机验收。

## 交互 TPOT 自由探索

`objective-exploration.json` 是自由探索的入口配置。累计模型 Token、请求、费用、墙钟和 Worker 用量计入账本。Worker 授权使用 `node_ms: null` 和 `jobs: null`，并单独登记每项作业授权。每段结束后保存检查点、加载活跃 Harness 并继续运行。`finish_exploration` 用于提名阶段候选，当前工作检查点、历史最佳和阶段提名分别登记。连续三段无成绩提升发出告警，探索继续。人工停止或安全熔断结束探索。基础设施持续故障触发暂停并写入通知和检查点。

新评分使用 `interactive-tpot-thinking-chat-v4`。W1/W2/W3/W4 权重依次为 20/70/5/5。`intake_public_chat.py` 读取 WildChat-1M 和 LongBench-E，保留完整对话或任务，用固定 Spark tokenizer 和对话模板开启思考。W1、W2 和 W4 采用原精确长度。W3 采用 3,990–4,202 token 的长文档任务，逐请求记录实际长度，主分使用配对原始 TTFT 的几何均值。独立观测器按模型的思考结束标记报告首个可见回答的时间。

`stage_chat_timing.py` 用登记的 E0 服务筛选开发和留出请求。W1、W2 和 W4 的 E0 输出需自然达到场景上限，W3 只核对流完整性。主集是由真实请求构成的受控性能负载，筛选会提高长回答请求的比例。自然泛化请求独立保留，省略 E0 输出长度筛选。启动单须提供通过 `repeat_e0.py plan --natural-eos --chat-workload … --chat-workload-sha256 …` 登记的六组同机 E0/E0 回执。主评分使用该负载自己的分场景门槛和波动。旧的固定输出与续写自然 EOS 使用各自的历史评分版本。[交互 TPOT 回放](./evidence/interactive-tpot-rescore-20261125.md)

`develop_measure` 复测同一引擎的输出真实性，允许 E0 与候选的 token 不同，并报告差异率及早停。W1、W2 按各请求真实生成的 token 计算 TPOT，W4 报告每路 TPOT、完成时间与总吞吐。经 E0 筛选的场景允许最多 10% 的平均输出长度和早停率差异。质量、不退化和配对精度门继续生效；驻留用量只观测，权重合规独立复核。`develop_stress` 保留固定输出压测。`stage_speed_bench.py` 单独登记 SPEED-Bench 专项数据及许可，结果不进入主评分。

允许自主选择投机解码、调度、融合、CUDA Graph 和其他保留 Target 契约的方案。当前授权的模型资产是固定 Spark GGUF，新增 Drafter 权重和训练需要单独登记。Spark 固定加载器的张量清单包含词嵌入、归一化和 28 层主模型权重，显式拒绝额外张量。快速质量路径分别观测数值、真实 Decode、EOS 终止协议和同邻居取消对照。完整随机采样只在历史最佳和最终提名触发：每阶段 α/2，只有不确定时允许第二次。有限样本仅覆盖已登记的采样事件；采样通过不等于 Q8_0 审计通过。

自由探索 Research Broker 使用 DeepSeek 官方 `POST /v1/tools/search` 和 `POST /v1/tools/fetch`。公开资料范围包括当前上游仓库、论文、PR、Issue 和网页。宿主保留请求、原始响应、文本分页、来源、版本和费用估算。预审索引由启动单选择是否挂载。环境保留容器隔离、密封数据隔离和外部资料不可信原则。Agent 可用 `research_reuse` 记录原创、技术迁移和上游补丁移植，交付包收录匹配改动路径的记录。独立 Search 和 Fetch 的公开价格均为每千次产生结果的成功调用 0.75 美元，查阅于 2026-11-25。[官方定价](https://api-docs.deepseek.com/guides/search/)

长操作可经 `submit_job` 开始，Agent 用 `job_status` 获取结果，期间可以操作本地工作区。多组确认可用 `job_cancel` 在一组结束后停止继续派发。Worker 按可信源码差异选择 Python 直接部署、Native 增量、Rust 重建或完整构建。Native 复用经过哈希和时间戳核验的 CMake 目录，按内存调节 Ninja 并发及 nvcc 线程。开发用 `develop_profile` 可选场景、窗口数和自定义 Token 序列，在独立容器内启动 Nsight。诊断及 Profiler 文件通过 `artifact_query` 分页。大型 Harness 文件可用 `harness_asset_save` 登记内容哈希，并在同一探索的后续会话用 `harness_asset_restore` 复取。

自由探索 Worker 将已接受的请求、作业 ID 和结果保存在独立于 SSH 会话的私有目录中。连接临时故障时按原作业 ID 重连。Worker 进程丢失或无法确认远端清理时保留最近的可信检查点和未结算记录。Harness 可通过 `dependency_fetch` 获取固定版本 npm 包，在 `deps/npm-lock.json` 登记依赖后于下一段载入隔离的只读依赖目录。

要从历史合法候选起步，在新工程授权的 `seed_candidate` 中登记 `engine` 和 `source: {path, sha256}`。从一次已有检查点继续时，用 `checkpoint: {path, sha256}` 指向 `run/checkpoint-exports` 中的导出。输入源码由固定 E0 的 SourceContract 重新核验，候选重新构建并通过质量检查和同期配对后才用作初始版本。三轮历史指标保留其原来的开发观测身份。

一次完整启动会写出 `execution-timeline.html` 和原始跨度数据。模型请求、Pi 工具、Worker 作业各有时间记录，组件构建还记录源码缓存准备、编译、验证和部署阶段。历史回执可用 `render_timeline.py RUN OUTPUT.html` 生成已有时间戳的图。

### 2026-11-25 · 同一 E0 的采样驻留对照（已停）

[本批结算](./evidence/thinking-chat-sampling-repair-settlement-20261125.json) 与[前批结算](./evidence/thinking-chat-controls-settlement-20261125.json)分开。前批没有完成采样；本批在 1,287.149 节点秒内完成 4 × 2,048 条采样，数值、自然 EOS、取消通过，但 E0 未通过驻留门。两个独立服务的 `ready` 均为 5,531 MiB；`post_requests` 在不含随机筛查的回放中为 5,533 MiB，含筛查时为 5,597 MiB。检查使用前者峰值 + 8 MiB，即 5,541 MiB，超出 56 MiB。已逐项复核两个阶段的服务 PID 读数；这些读数不能归因到某类分配或权重镜像。回执的 `sampling_screen.status=reference` 只标记样本来自 E0，整体为 `violation`，并未接纳为候选参照。

独立自然输出回放、历史 Python 候选和两个 `mmq.cuh` 候选均未执行。含 Graph 回放的已结算节点时间为 1,683.974 秒，付费调用为零。结算时无 Level 3 容器或 GPU 计算进程，共享锁可取得。未重试、未调大驻留容差、未启动探索或正式 W。后续应预先登记同负载 E0 显存基线与独立复核，再检验现有驻留负控；本次差值本身不是可用来放宽门槛的校准值。

## Level 3 六域 v2 冻结登记（2026-11-24；以下为当时状态）

[09 Level3 自由探索最终方案](../../../研究方案/09%20Level3%20自由探索最终方案.md) 是本轮协议正本。`acceptance-v2.json` 冻结六个能力域的 15 项必要检查，SHA-256 为 `4c029badae73d820f2a0a1175dfb4bbf3e1426e7b6e1919c7bf5b0ce06e56c20`。从仓库根目录运行 `python experiments/spark26_gguf/level3/audit_registration.py` 检查登记和引用完整性；它不执行评测、不签收能力，也不授予启动权限。

| 域 | 基准状态 | 复用 / 当前主要缺口 | 首次探索 |
|---|---|---|---|
| A · Runtime | 部分实现 | 复用 Pi / Flash Max 长 SSE 与 1,205 秒静默回执；新入口、异常和跨段记账仍要集成验证 | 必须 |
| B · Research | 未实现 | 已有隔离和历史环境锁，不是 Research 接口；新增公开搜索、正文、指定 Commit、真实环境发现和依赖隔离 | 必须 |
| C · Engine Development | 部分实现 | 已有固定 E0 与单文件 Worker；需全树、原生构建部署、版本分支、可编程诊断和实测 Profiler | 必须 |
| D · Trusted Evaluation | 部分实现 | 复用窄范围 G0/G1 和 W1–W4；需 Native 正负控、实测 E0 噪声和独立 Q/T | 必须 |
| E · Harness Evolution | 部分实现 | 复用 Harness 仓和上轮隔离预演；需正常换段、WIP / 可信事实分离和从新 Pi 的实际调用证据 | 必须 |
| F · Governance | 部分实现 | 复用 API/GPU 窄路径账本；需全链账本、来源、紧急停止和真实资源授权 | 必须 |

登记冻结时，本地 `main` 与从 GitHub 实际查询的 `origin/main` 同为 `ae0b11e033186c5fe88c92414f09c16d19d0e535`；本地 `.serena/` 不属于协议。登记中的当前状态是基准快照，后续必须由独立证据更新，不修改冻结文件来表示进度。新接口代码和模拟覆盖不能替代 B/C/D/F 所要求的网络、实机或授权证据。

## 启动前补充协议

[补充协议 v1](./PROTOCOL-ADDENDUM-v1.md) 登记一次有界恢复、E0 上游来源边界、分账、数值风险、传输证据和留出集共审。
它和冻结的六域清单一同使用，不改写原清单；未接入真实执行路径的约束不能视为通过。

## 生命周期离线实现

`task-v1.txt` 从协议第 5 节提取，SHA-256 为 `af434c1f192c63c6da33f7757bfcfc5b2e81dd7808d9d4613f36d514972a6c86`。这是目标输入，不是 Runner 已提供这些工具的证明。实际系统提示必须由已注册工具和现场约束生成；未验收时不得向 Agent 宣称相应能力可用。

`evolveengine/v3_level3_lifecycle.py` 使用可信宿主的 Journal，分别登记已绑定 Engine/Build 的质量通过、可信开发测量、检查点、正常换段、异常恢复和最终交付。检查点分别引用 E、H、M_E、M_H、R、WIP 的内容地址，恢复逐项复核；WIP 绑定基准 Engine，不进入最终交付。该模块由外层调用，不是 RPC 或调度器；还需连接真实 artifact 验证、Harness active pointer、清理与结算证据、实时预算、Pi 加载和调用证据。现有夹具测试不签收 A2/E1/E2 或 F 域。

## 当前运行入口

- `rehearse_runtime.py NEW_DIR --archive PINNED_E0.tar.gz`：真实 Pi、完整 E0 文件树、模拟 SSE 与模拟 Worker；检验多语言候选上传、激活 H1、正常换段、一次截断恢复和 WIP 隔离。`--scenario second-truncation` 验证第二次截断终止。回执不用于签收真实构建、GPU 或性能。
- `preflight_live.py --execute-authorized-probe NEW_DIR`：仅经授权使用宿主的 `DEEPSEEK_API_KEY`，三个有界合成工具轮次，无 Worker、无 GPU；不启动自由探索。
- `node_preflight.py inventory NEW_DIR`：只在登记的 `spark-c3bc` 执行。读取 GB10、Docker、设备占用、已有 Level 3 容器、映射读取权限和镜像身份；不拉镜像、不启动容器、不使用 API。`pins` 在本地生成开发 Judge 和 W1–W4 客户端的 LF 字节哈希。
- `node_preflight.py baseline NEW_DIR --registration CONFIG --sha256 PIN`：可信节点上的 E0/E0 工程预检。核对同一 boot/GPU、历史共享锁、镜像、源码、GGUF 及开发参考后，两次分别冷构建、采集质量和 W1–W4。不通过时不出分；不打开留出集，不调用 Agent，不替代 Native 正负控或 Profiler 实机验收。
- `v3_level3_research.py`：只搜索和读取宿主预审、冻结的公开本地索引；SGLang 文件必须在受信 Git 的 E0/祖先提交中逐字节验证。没有 Agent 任意 URL 抓取接口；公开在线采集和实验室环境发现尚未完成验收。

- `v3_level3_holdout.py`：主持方可以在第一个会话前登记共审元数据的可信哈希；只有记录并核验 Agent 的最终交付后，独立保管方才可一次性解封。无 Agent RPC；当前未登记真实共审文件，也未启封或调用共审模型。

以上入口使用既有锁定的 Pi/Bridge 镜像、离线 tokenizer 和测试依赖。`launch_live.py` 的工程探索使用已验证的同配置 Graph E0 回执和真实 Worker。Worker 核验并复用 E0 wheel，重新测量 E0，按同一套质量、驻留和真实文本配对规则评估候选。驻留门按同配置 E0 观测峰值加 8 MiB 暂定。重复性、正负控及设备追踪的完整实机验收，在真实探索的首个 DeepSeek 请求被上游接受并写出 `deepseek-started.json` 后开展。正式验收和 Q/T 启封继续使用原审批及保管流程。

## Spark 实机预检记录（2026-11-24，香港时间）

- 登记节点已通过可信节点 `inventory` 预检；旧窄任务 Worker 的 relay 不参与这条链路。站点的 SSH 连线资料由运行方单独保管，不放入仓库。
- 实测 Docker `29.6.2` 的镜像 inspect 可省略 `Config.Volumes`；兼容缺省字段后，原锁定 ARM64 运行镜像通过核对，声明匿名卷的镜像仍被拒绝。持现有 `/home/jiazheng/.evolve-spark26-gpu.lock` 的短时探测通过，不按调用账号的 HOME 创建另一把锁。自有 `/proc/.../map_files` 探测通过（不代表服务映射已核验）。当时无计算进程、清理隔离标志或残留 Level 3 容器。E0 源码包、Q8_0 权重、模型配套文件和开发质量锚点均已复核哈希。
- `nvidia-smi` 的 `memory.used` 实测为 `[N/A]`。预检如实记为未观测，不能用 `0` 或系统总内存代替。Native 冷构建镜像尚无已核验的固定 digest，完整 E0/E0 登记也未完成。没有构建、启动容器、运行 GPU 工作负载、调用 API 或打开留出集。
- 复测原始预检记录：`spark-c3bc:/var/tmp/ee-l3-inventory.WVd7q9Eu/report/preflight.json`，SHA-256 `109226bce6c28dde26b4ef95678bbc33c7f3640cde290f4877db9c24a4055895`。

## Native 构建与 E0 工程验证（2026-11-24，香港时间）

- 设备总量 `[N/A]` 仍保留。`cudaMemGetInfo` 校准未通过（CPU 对照扰动了读数），cgroup `memory.current` 也不能证明 CUDA 分配量。改用 NVML 的**已归属 compute PID** 读数：CPU-only 128 MiB 对照和 CUDA 12/16/128/512 MiB 各 3 次，共 15/15；每次 CUDA 增量和回落均符合预登记。最小已校准增量是 12 MiB，不能排除更小镜像、未采样瞬态或未计入这些 PID 的内存。
- 原始报告 `spark-c3bc:/var/tmp/ee-l3-tools.f904ed62/nvml-report`，`result.json` SHA-256 `1b0bd8ee856d3dce12873abf93e9c4389449b5506de9f8c7fcd5be9d9a5f3884`；失败的 CUDA 全局可用内存报告 `spark-c3bc:/var/tmp/ee-l3-meter.Tdc6Ztvk/report/result.json`，SHA-256 `1542c3bf68aee543870f8d57bf9d71b412372ca83d434b9c3662c13312bd4202`。E0 开跑前重放 NVML 原始样本，绑定同节点、boot、GPU UUID 和驱动；采集时从 Docker 和 `/proc` 核实服务 PID。质量客户端先等待可信宿主完成 ready 样本才发首个请求。此观察不是候选驻留阈值；要用同配置 E0 和代表性 BF16 驻留负控另行签收。
- 固定 ARM64 Native 构建镜像 ID：`sha256:ac63eb5e39f5dc44f65c7b428ba6759e56ae2ecaa8cfea6b06aeadf2ee5ea9b2`。三个 CMake 归档与构建 wheel 固定哈希，两套 E0 Cargo 锁文件合计 585 个 registry 依赖；离线验证两套 Cargo metadata。实机无网络、非 root 冒烟验证 Rust/Cargo 1.92.0、PyTorch 2.14.1+cu130 和依赖可读；这不是 E0 编译成功或某个 CUDA 路径已执行的证明。
- 首次 E0 工程运行 `spark-c3bc:/var/tmp/ee-l3-e0e0.f38c`：E0a 冷编译在 271.587 秒因 Docker cgroup 内存上限退出（10 jobs、32 GiB；内核记录 `cicc` 被杀），未启动质量或性能作业。独立的新登记 `6019167505bef1aca05a365c08c636f9c666d2a343c4e3ed3a32f4c48a729947`（6 jobs、80 GiB、冷构建不重用缓存）已在 `spark-c3bc:/var/tmp/ee-l3-e0e0.61e7` 启动。E0a 的 Native 和 Engine wheel 均已构建（1,767.764 秒），随后质量观测器因非 root 无法读取入口而停。该轮没有质量通过或性能结果。只在两臂分别核验质量和 W1–W4 后写配对结果；单对不据以冻结噪声带。两次都没有 Agent 或付费 API，也不启封 Q/T 或启动 W。 `61e7` 固定的是此前的工程吞吐摘要，不能当作 P3 分场景目标分数。后续登记同时保留 TTFT、TPOT 和总体吞吐，并固定 W1/W2 为解码吞吐、W3 为 TTFT 倒数、W4 为整波吞吐，按各场景的 E0 比值合成。

- `native_controls.py` 从固定 E0 生成三份可信操作端夹具：计算等价、可用 GPU trace 区分的新 MMVQ 内核符号；实际改写 K/V 形状对应的 Q8_0 MMVQ 结果的故障；以及不参与计算的 BF16 驻留镜像。夹具只通过源码重放测试，未做硬件判定，不放进 Agent 上下文。`v3_level3_worker_channel.py` 的候选、构建父回执和租约绑定已通过本地传输测试，未作为真实 Agent 数据通道上线。

- 只开放观测器只读 bind 根的读取权限，可信证据父目录继续保持私有。复用 `61e7` 的哈希绑定 wheel 进行观测器诊断（`/var/tmp/ee-l3-observe-replay.r04`，不计 E0 配对）；确认运行时找到了 wheel 的 SM100 Native 库，但因 Torch CUDA 符号 ABI 不同而无法加载。两镜像都带有两套 Torch。编译时使用了系统 Torch ABI，运行固定在 `/opt/spark26-deps/torch`，不能靠版本标签或 Python import 证明编译兼容。
- 构建器现显式固定 `Torch_DIR` 到运行时 Torch，同时固定子进程的 `PYTHONPATH`。非 root、无网络 ARM64 的 CMake 实测确认用了对应头文件和库。新 E0/E0 运行 `/var/tmp/ee-l3-e0e0.abi-c6`（登记 `6da20e5b53ad2d83054d79c9d7a564299a1cd72621dbc5975f2f7455378de088`）使用独立冷构建；实际构建和回放结果见下。
- `abi-c6` 的 E0a 在 1,748.424 秒产出两组 ABI 已对齐的 wheel；后续在不可执行的 `/tmp` 上加载 Triton 生成库失败。`replay.c7` 把 Triton 指向专用 tmpfs 后又定位了 SGLang 自身的 JIT 路径，同样不是质量拒绝。两次均留存原始错误，不出分。
- `spark-c3bc:/var/tmp/ee-l3-replay.c8` 重新哈希并复用 `abi-c6` wheel，专用临时 JIT 挂载保持 `nosuid,nodev`，常规 `/tmp` 保持 `noexec`。质量三组检查通过（83.337 节点秒，采样峰值 4,774,166,528 字节），非剖析 W1–W4 全部取得计时（100.856 节点秒）。汇总 SHA-256 `60585dfabd146cf4c0459e2357d8445d93411ebb1a70f75cb7e3e74277408559`。这是修复重放，不是两个独立冷构建的 E0/E0，也没有观察优化增益。
- 在 `spark-c3bc:/var/tmp/ee-l3-e0e0.v4` 启动新独立配对；登记 SHA-256 `fe1971587d814d190a6e2f1dd8cd22e6f8197d6b86d362df746258786a4a06ed` 固定实际 Worker 端实现。首臂完成冷构建（1,711.240 节点秒）、质量（83.971 节点秒）及 W1–W4（101.765 节点秒），第二臂已启动独立冷构建。Native 控制和六组反向平衡的 E0/E0 计时重复已条件排队；前一步失败、缺回执、采样不稳或隔离异常不能默默跳过，不自动重试。重复包络是工程描述值，不是置信区间或正式 W 的阈值。
- 真实候选的 Worker 入口只有在受信授权绑定同一基准登记、Native 控制计划和完整结果，且独立计算与驻留负控实测按指定路径失败后，才采用校准的驻留上限；否则仅可走 E0 工程路径或停止。此入口不开放 Profiler 或 Q/T。回执篡改拒绝已在 Windows 和 Linux 通过，真实二进制 SSH 往返及实际候选调度仍待验收。
- `profile_control.py` 接入独立的、非计分的固定对照采集；记录同一 wheel 的实际映射、CUDA API/Kernel 时间线和 GPU 上执行的指定符号，不靠库里出现符号来证明执行。Nsight 2026.4.1 在锁定镜像内以非 root、无网络、无 GPU 的方式实测生成并导出了报告；真实 CUDA trace 仍待回执。固定 Native 正控的采集已条件排队，无任意候选的可信 Profiler 授权。
- `diagnose_e0.py` 和 `diagnostic_stop_control.py` 已部署：固定离线 E0 wheel 下运行独立的非计分 CUDA 脚本，不挂载权重、Judge、网络或 Agent 状态。停止对照需先真实观测到该容器的 CUDA 进程，再发送一次 SIGINT，验证容器与 GPU 进程清理并保留保守扣账。两项尚待实机回执；这不是整个 F2 的签收。

## 历史登记：v1（仅保留追溯）

审计基准：`ae0b11e033186c5fe88c92414f09c16d19d0e535`。2026-11-24 核对本地 HEAD、GitHub `origin/main` 和工作树，确认基准相同，仅有未跟踪的本地 `.serena/`。`acceptance-v1.json` 在 Level 3 实现前冻结。具体 Native 正、负控及评分覆盖、隔离、工具链和测量协议必须在相应实机试验之前单独登记。没有实测回执的项目保持未验收；不能根据该清单的存在认定已完成。

| 工程线 | 基准已有 | 缺口 / 完成证据 |
|---|---|---|
| A Agent Runtime | 长回复、真实 Pi + 模拟上游、费用门禁、分页读取、一次恢复 | 新 Level 3 入口/配置；集成测试验证真实运行时不再对准一个函数；复用历史长 SSE 和静默实验，不重复付费 |
| B Engine 与 Worker | E0 诊断镜像有固定上游源档和依赖锁；当前开发 Worker 只替换固定的 `gguf.py` | 完整源码快照、跨文件候选、构建输入/产物绑定、独立构建与缓存、实际修改的 Native 路径执行；诊断 E0 不是原生重建证明 |
| C Judge/Profiler | G0/G1、四场景测量、进程内 Q8_0 驻留检查适用于先导范围 | 多文件与已加载产物绑定、独立同条件 E0、代表性 Native 负控、外部观测、Profiler 实机回执；不声称完备检测任意原生代码 |
| D 探索/成本 | 窄任务提示；第一次显式交付停止；异常恢复可加载已激活 H | 双目标、正常换段、可信检查点与失败事实、独立最终交付；跨段计量、最终独立配对、选择与历史最佳分开、可复现结果包 |

验收不要求获得正收益、创造 H1 或使用某种指定优化手法；能力测试本身必须真正使用 Native 路径和下一会话工具。授权和所有必要门禁均满足后才能启动完整 Level 3；不启动 W 或 B0/B2。不改变 `v3_dev_rpc` 窄任务的配置和历史回执。
