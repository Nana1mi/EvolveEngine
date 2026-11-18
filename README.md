# EvolveEngine

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

`journal.sqlite3` 保存内容寻址的 E/H/M、原始回执、操作账本与不可改写的事件。`report.json` 导出运行清单、资源、D/C 建库开销、策略晋级记录、封存版本及评估结果。D 分叉费用归 Evolving 主链，共享首轮候选池有两份逻辑扣费和一份物理执行。C 续跑与最终 Q/T 在主链封存后运行。恢复时无回执的操作以 `uncertain` 结算预留额度。

本地测试覆盖规则回放、拒绝篡改候选、引擎门禁、数据区访问、D 配对晋级以及中断恢复。模拟吞吐来自确定性测试夹具。DGX Spark 的 CUDA/SM121、Pi 工具调用与真实模型结果由 P1 实机联调获取。

## P1 实机配置与单节点联调

控制器运行在普通 CPU 主机。Spark 端用 Python 3.10+、NumPy、Git、Docker 和 NVIDIA Container Toolkit。配置 Spark 的 SSH 主机别名，并把同一版本的 EvolveEngine 和锁定提交的 mini-sglang 部署到各节点。镜像需要预装兼容 GB10 / SM121 / aarch64 的 CUDA、PyTorch、Triton、mini-sglang 所需 Kernel 和 Transformers。各节点使用同一份已下载的 Qwen3-0.6B 权重和 tokenizer。

- 从 worker 模板派生各 Spark 共用的私有配置，填入镜像 digest、引擎 commit、显式 attention backend、模型目录、profile 请求和各数据区不同的自然 EOS 例子。用锁定 tokenizer 生成输入 token IDs。每个数据区至少有一个参考模型会在上限前生成 EOS 的请求。 [Worker 配置模板](configs/spark.worker.example.json)
- 从控制器模板派生私有配置，填入节点别名、本地引擎克隆、远程目录、worker 配置哈希和请求词表哈希。两个配置均由实验者保管，lease 文件需要可写。 [控制器配置模板](configs/spark.controller.example.json)
- 控制器安装 Pi 1.5.2，并向控制器进程提供 `DEEPSEEK_API_KEY` 环境变量。示例的 `deepseek-flash` 是 DeepSeek 官方的 V4.1 Flash 调用 ID。先导记录试调用实际返回的模型与可用指纹。正式实验锁定可验证的固定版本或响应指纹。 [DeepSeek 更新记录](https://api-docs.deepseek.com/updates/) · [Pi 自定义模型配置](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models-configuration.md)

```sh
npm install -g @earendil-works/pi-coding-agent@1.5.2
python -m evolveengine.spark_worker --hash-model /opt/models/qwen3-0.6b-pinned
python -m evolveengine hash-config .evolveengine/spark-worker.json
python -m evolveengine make-token-ids --tokenizer /path/to/qwen/tokenizer.json --output .evolveengine/token-ids.json
python -m evolveengine new-seal-key .evolveengine/seal.key
python -m evolveengine preflight --config .evolveengine/controller.json --salt-file .evolveengine/seal.key --node spark-1 --output .evolveengine/one-node-preflight.json
python -m evolveengine optimize --config .evolveengine/controller.json --salt-file .evolveengine/seal.key --node spark-1 --preflight-receipt .evolveengine/one-node-preflight.json --steps 2 --output .evolveengine/p1
```

在权重所在主机执行模型哈希命令。预检核对 Pi 版本、DeepSeek 元数据与实际用量，以及 Spark UUID、镜像、模型资产、运行时版本、独立参考质量门和 E0 两次测量。更换镜像、模型、Kernel 或协议参数后重新预检。

真实优化路径通过 Pi 的 `read`、`edit` 和 Linux 上受限的 `bash` 实现单个候选。工具只操作固定 Git 快照中允许的源码。实现后独立请求同一模型审计实际 diff，审计开销计入实现阶段。最后由程序执行文件范围和正确性门禁。

Spark 测量使用 mini-sglang 原生 LLM 路径。客户端每次提交一条 token-ID 请求，计时并校验全部输出。TTFT 和 TPOT 来自 token 回调时间。GPU 活动间隙作为候选排序特征。质量门另启独立的 Transformers 参考进程，比对真实解码路径上的 EOS、完整词表、选定时点的 KL 和位置一致性、固定长度及交错请求。示例采用小规模质量样本用于联调，正式样本量和阈值由先导确定。 [mini-sglang 原生 LLM 接口](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/llm/llm.py)

## P2 先导与独立对照

四个节点预检全部通过后运行一对。D/C 建库和终评单列成本，D 的策略续跑计入对应 Evolving 组。每台 Spark 通过本地租约独占加速器。同一检查点的两臂复用首轮输入，并各自结算逻辑费用。

```sh
python -m evolveengine preflight --config .evolveengine/controller.json --salt-file .evolveengine/seal.key --output .evolveengine/four-node-preflight.json
python -m evolveengine pilot --config .evolveengine/controller.json --salt-file .evolveengine/seal.key --preflight-receipt .evolveengine/four-node-preflight.json --replicate 0 --output .evolveengine/pilot-0
python -m evolveengine summarize .evolveengine/pilot-0/report.json
```

默认在一个 C 检查点做先导配对续跑。`--causal-points 3` 使用早、中、晚三个检查点，`--ablation` 加跑关闭跨会话 Memory 的两组。增加独立 `--replicate` 编号和新输出目录会轮换节点职责与执行顺序。`summarize` 将同谱系的 C 结果聚合为一个统计单位，给出每对对数差、区间和失败数。

已验证 P0 闭环、真实 Pi CLI 对本地流式 API 的计量和工具限制、mini-sglang Git 快照及数值比较函数。Spark 预检、真实 DeepSeek 请求、CUDA Kernel、权重执行与多节点先导需要目标环境实测。

```sh
# 可选。使用刚安装的真实 Pi 验证工具限制和流式计量。
EVOLVE_PI_TEST_COMMAND='["pi"]' python -m unittest discover -s tests -v
```
