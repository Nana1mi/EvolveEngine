# EvolveEngine

**协议正本：** [研究方案索引](%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88/README.md) · [当前 v2.2 已定决策（§8）](%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88/05%20%E6%A1%86%E6%9E%B6%20v2%20%E8%AE%BE%E8%AE%A1.md)。下文的固定组/进化组主链是 v1；冻结策略的 RQ-A 有独立入口。

EvolveEngine 比较两种推理引擎优化方法。两组从共同的 E0 和 H0 起步，在相同四维预算内累计合格引擎补丁。Evolving 组还可以用自己的预算从轨迹中学习候选排序规则，并在独立开发分叉上进行验证。

研究背景见 [研究总方案](./%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88/00%20%E7%A0%94%E7%A9%B6%E6%80%BB%E6%96%B9%E6%A1%88.md)。本仓库实施路线使用单节点 DGX Spark 上的 mini-sglang 与 Qwen3-0.6B BF16。Pi 的生成、实施与更新阶段使用同一配置的 DeepSeek 模型。

## P0 离线验收

在仓库根目录执行。Python 3.10 或更新版本即可运行模拟和单元测试。

```sh
python -m unittest discover -s tests -v
python -m evolveengine mock --output .evolveengine/accept --ablation
python -m evolveengine mock --output .evolveengine/reject --scenario reject
python -m evolveengine mock --output .evolveengine/deviation --scenario deviation
python -m evolveengine mock --output .evolveengine/quality-failure --scenario quality_failure
```

恢复验证的第一条命令在持久化预算预留后退出，退出码为 75。第二条命令复用同一个目录，读取原来的日志和快照并继续。

```sh
python -m evolveengine mock --output .evolveengine/resume --interrupt-at replicate-0/main/evolving/step-0/implement
python -m evolveengine mock --output .evolveengine/resume
```

`journal.sqlite3` 保存内容寻址的 E/H/M、原始回执、操作账本与不可改写的事件。`report.json` 导出运行清单、资源、D/C 建库开销、策略晋级记录、封存版本及评估结果。D 分叉费用归 Evolving 主链；C 续跑与独立 Q/T 裁判在主链封存后运行。

- 排序规则可在 `unknown` 观测下触发，也可把指定动作跨类别提前；无命中时完全沿用 H0。规则不增删候选。
- 仅在同一 D/C 配对的同一步共享工作：同选同一候选，共享实现、审计、门禁和筛选/复测；首个不同选择立即分开，之后不重新合并。两臂及共同外层账户按原上限预留，两份逻辑账原子结算、只实付一次；中断而无回执则均按预留记 `uncertain`。RQ1 主链和独立 A/A 噪声校准不共用吞吐读数。
- D 上相同交付物通过 D 门禁后记 `log_gain: 0.0`，`rates` 为 `null` 值而非虚构吞吐；门禁失败不能记作合法零。C 终评在该比较内按引擎共用独立 Q/T 回执；即使与参照相同也要取得真实的质量、性能和延迟回执。运行清单的 `harness_revision` 区分新旧算法：请使用新的输出目录，旧封包用原版本裁判，不跨版本合并报告。

- Memory 开关只控制经验是否可见；无论开关如何，生成器和过滤器都使用**本臂当前父引擎**已试的 `(动作, 目标范围)` 排除表，不含结果。有效响应过滤后为空时最多再生成 **1 次**，不重跑 profile、每次照常预留和付费；第二次仍空则 `empty_pool`，无余额则 `budget_stopped`。每次抽取在 `draw-*` 区分原始数量、越界、池内重复、历史重复和可选数量，回放按决策而非抽取次数计。
- 两条主链按各自墙钟预算的 **0、1/4、1/2、3/4、1** 保存当时已签入的引擎。`submission.<arm>.budget_points` 可回查签入事件。封卷后 `evaluate` 先完成预定 Q/T，再在研究评估账上使用同一公开 8 请求块重测各快照与 E0。重复哈希在这条次要曲线内共用实测；无有效参照或余额不足记缺失。`S_dev.points` 是重测曲线，`process_metrics.<arm>.screening_trace` 只是被选中补丁的筛选读数，不能代替它。新封包格式为 `evolveengine.sealed-pair.v2`。

针对上述反例单独运行：`python -m unittest discover -s tests -p test_paired_invariants.py -v`。

本地测试覆盖规则回放、拒绝篡改候选、引擎门禁、数据区访问、D 配对晋级以及中断恢复。模拟吞吐来自确定性测试夹具。DGX Spark 的 CUDA/SM121、Pi 工具调用与真实模型结果由 P1 实机联调获取。

## v2.2 RQ-A：只测一份冻结策略，不在评估谱系上学习

`mock-rqa` 与 v1 `mock`/`pilot` 是两个不同实验。下面用固定 H0 作为 H* 的离线阴性对照，两个谱系都应精确为零：

```sh
python -m evolveengine mock-rqa --frozen-policy configs/rqa.frozen-h0.example.json --replicate 0 --output .evolveengine/rqa-0
python -m evolveengine mock-rqa --frozen-policy configs/rqa.frozen-h0.example.json --replicate 1 --output .evolveengine/rqa-1
python -m evolveengine summarize-rqa .evolveengine/rqa-0/report.json .evolveengine/rqa-1/report.json
```

- 母链只用 H0，预先锁定 H*。在母链预算的 50%、75% 取不晚于边界的最后一个完整 `/finished`（含失败的经验）；若尚无完成步骤就取初态。记录完整 checkpoint、墙钟和来源事件；配合搜索 journal 可以恢复 E、M 及其排除表。两个续跑各自从母链出发。这**不是** v1 `S_dev` 的最佳引擎签入曲线，也不是按步数抽取的 v1 D/C 库。
- `seal-rqa` 需要 `--frozen-policy`、已通过的 preflight 和独立终评密钥承诺（其余参数见 `--help`）。搜索机无最终 Q/T 密钥；把封包和预登记摘要交独立的 `evaluate`，用 `summarize-rqa` 汇总裁判输出的 `sealed_evaluation.json`。该入口**不生成或验证训练谱系来源**：正式使用前仍须独立冻结训练产物及留出谱系清单，不可把先导结果当作确认性证据。
- 各点用同一盲测块上实际**交付**的输出 token/s 算对数比（某臂回退时使用实测的合法起点），不是两份 V1 归一化加速比之比。两点先等权平均，独立母链才计一个样本。双侧 95% 区间下限 > 0 判正；3% 单列。协议/冻结策略不同、重复谱系或少一个预定点均不能混报；有点准入、起点终评或计时未完成则显式列缺，不拿另一点重加权、不作正式通过判断。模拟结果永不计作实机确认。

## P1 实机配置与单节点联调

控制器运行在普通 CPU 主机。Spark 端用 Python 3.10+、NumPy、Git、Docker 和 NVIDIA Container Toolkit。配置 Spark 的 SSH 主机别名，并把同一版本的 EvolveEngine 和锁定提交的 mini-sglang 部署到各节点。镜像需要预装兼容 GB10 / SM121 / aarch64 的 CUDA、PyTorch、Triton、mini-sglang 所需 Kernel 和 Transformers。各节点使用同一份已下载的 Qwen3-0.6B 权重和 tokenizer。

单节点联调、预检与 v2 RQ-A 不要求四台：在私有配置的 `nodes` 只列实际可用的 SSH 主机，支持一台或两台；不要为同一 GPU 配多个别名。旧 v1 `pilot` 仍须四个独立节点，不能用双节点配置冒充。SSH 须能以 `BatchMode=yes` 认证；能手工输入密码登录不等于控制器可运行。

- 从 worker 模板派生各 Spark 共用的私有配置，填入镜像 digest、引擎 commit、显式 attention backend、模型目录、profile 请求和各数据区不同的自然 EOS 例子。用锁定 tokenizer 生成输入 token IDs。每个数据区至少有一个参考模型会在上限前生成 EOS 的请求。 [Worker 配置模板](configs/spark.worker.example.json)
- 从控制器模板派生私有配置，填入节点别名、本地引擎克隆、远程目录、worker 配置哈希和请求词表哈希。两个配置均由实验者保管，lease 文件需要可写。 [控制器配置模板](configs/spark.controller.example.json)
- 控制器安装 Pi 1.5.2，并向控制器进程提供 `DEEPSEEK_API_KEY` 环境变量。示例的 `deepseek-flash` 是 DeepSeek 官方的 V4.1 Flash 调用 ID。先导记录试调用实际返回的模型与可用指纹。正式实验锁定可验证的固定版本或响应指纹。 [DeepSeek 更新记录](https://api-docs.deepseek.com/updates/) · [Pi 自定义模型配置](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models-configuration.md)

```sh
npm install -g @earendil-works/pi-coding-agent@1.5.2
python -m evolveengine.spark_worker --hash-model /opt/models/qwen3-0.6b-pinned
python -m evolveengine hash-config .evolveengine/spark-worker.json
python -m evolveengine make-token-ids --tokenizer /path/to/qwen/tokenizer.json --output .evolveengine/token-ids.json
python -m evolveengine new-seal-key .evolveengine/development.key
python -m evolveengine preflight --config .evolveengine/controller.json --salt-file .evolveengine/development.key --node spark-1 --output .evolveengine/one-node-preflight.json
python -m evolveengine optimize --config .evolveengine/controller.json --salt-file .evolveengine/development.key --node spark-1 --preflight-receipt .evolveengine/one-node-preflight.json --steps 2 --output .evolveengine/p1
```

在权重所在主机执行模型哈希命令。若控制器在 Windows 上，可用 `["node","固定版本 Pi 的 dist/bundle/cli.js 绝对路径"]` 配置 `pi_command`。预检核对 Pi 版本、DeepSeek 元数据与实际用量，以及 Spark UUID、镜像、模型资产、运行时版本、独立参考质量门和 Profiler。每台 Spark 都用同样的公开 E0 请求形状，分别重复运行 8 请求搜索块、16 请求晋级块和正式计时块。每种规格以各自的实际超时运行，报告吞吐波动、含启动的总耗时和削短响应、截断 logits 的拒绝结果。预检报告也给出按配置上限估算的一步预算与策略验证预算。更换环境或协议参数后重新预检。

真实优化路径通过 Pi 的 `read`、`edit` 和 Linux 上受限的 `bash` 实现单个候选。工具只操作固定 Git 快照中允许的源码。实现后独立请求同一模型审计实际 diff，审计开销计入实现阶段。最后由程序执行文件范围和正确性门禁。

Spark 测量使用 mini-sglang 原生 LLM 路径。客户端每次提交一条 token-ID 请求，计时并校验全部输出。TTFT 和 TPOT 来自 token 回调时间。GPU 活动间隙作为候选排序特征。质量门另启独立的 Transformers 参考进程，比对真实解码路径上的 EOS、完整词表、选定时点的 KL 和位置一致性、固定长度及交错请求。示例采用小规模质量样本用于联调，正式样本量和阈值由先导确定。 [mini-sglang 原生 LLM 接口](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/llm/llm.py)

## P2 先导与独立对照

四个节点预检全部通过后运行一对。D/C 建库和终评单列成本，D 的策略续跑计入对应 Evolving 组。每台 Spark 通过本地租约独占加速器。同一检查点的两臂复用首轮输入，并各自结算逻辑费用。

裁判先在自己的权限范围内生成 Q/T 私钥，把命令输出的 `commitment` 登记为 `FINAL_KEY_COMMITMENT`。开发控制器只接收这个哈希。

```sh
python -m evolveengine new-seal-key .evolveengine/judge/qt.key
```

```sh
python -m evolveengine preflight --config .evolveengine/controller.json --salt-file .evolveengine/development.key --output .evolveengine/four-node-preflight.json
python -m evolveengine pilot --config .evolveengine/controller.json --salt-file .evolveengine/development.key --final-commitment "$FINAL_KEY_COMMITMENT" --preflight-receipt .evolveengine/four-node-preflight.json --replicate 0 --output .evolveengine/pilot-0
```

`pilot` 先封存主链、C 续跑和可选消融提交，写出 `sealed_submission.json` 和开发阶段的 `report.json`。将封存包传给裁判，并独立登记命令输出的 `submission_digest` 为 `SUBMISSION_DIGEST`。裁判使用相同的控制器配置和 token 清单、原先登记的提交哈希，以及私有 Q/T 密钥。裁判进程只运行质量门与计时，不需要开发账本、Pi 或 DeepSeek 凭据。

```sh
python -m evolveengine evaluate --config .evolveengine/controller.json --submission .evolveengine/pilot-0/sealed_submission.json --submission-digest "$SUBMISSION_DIGEST" --final-salt-file .evolveengine/judge/qt.key --output .evolveengine/judge/pilot-0
python -m evolveengine summarize .evolveengine/judge/pilot-0/sealed_evaluation.json
```

正式批次应在所有预定交付物锁定后统一运行 `evaluate`。预算点只增加公开 `S_dev` 测量对象；中间快照不因此获得正式 Q/T 评分，也不返回优化或晋级。

默认在一个 C 检查点做先导配对续跑。`--causal-points 3` 使用早、中、晚三个检查点，`--ablation` 加跑关闭跨会话 Memory 的两组。增加独立 `--replicate` 编号和新输出目录会轮换节点职责与执行顺序。开发报告和裁判报告都记录同一个提交哈希。裁判报告含吞吐及 TTFT/TPOT 分位数。`summarize` 将同谱系的 C 结果聚合为一个统计单位，给出每对对数差、区间和失败数。

已验证 P0 闭环、真实 Pi CLI 对本地流式 API 的计量和工具限制、mini-sglang Git 快照及数值比较函数。Spark 预检、真实 DeepSeek 请求、CUDA Kernel、权重执行与多节点先导需要目标环境实测。

```sh
# 可选。使用刚安装的真实 Pi 验证工具限制和流式计量。
EVOLVE_PI_TEST_COMMAND='["pi"]' python -m unittest discover -s tests -v
```
