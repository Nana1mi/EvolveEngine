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
