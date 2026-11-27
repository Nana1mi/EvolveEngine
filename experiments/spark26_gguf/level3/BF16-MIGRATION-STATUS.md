# Spark-X2.6-1.7B BF16 Migration & Readiness Status

## 1. Official BF16 Weight Provenance & Verification

- **Source**: Hugging Face `XHToken/Spark-X2.6-1.7B` (local node mirror commit `e6406af83988f26e87ea92455f31db3f6476131b`, verified 100% byte-identical against latest HF `0e60633b345b19ba9f56ed1fda197acd271e8546`, where only README/SVG changed).
- **Controlled Asset Directory on `spark-c3bc`**: `/var/tmp/ee-l3-bf16-assets/model` (13 files, read-only `0444`).
- **Tensor Audit**: 226 tensors, 100% `BF16` (`1,707,657,216` params / `3,415,314,432` bytes), `tie_word_embeddings=true`, 28 layers (`21` sliding-window attention `window=512` + `7` full attention), **no MTP/nextn/draft weights**.
- **Key File SHA-256**:
  - `model-00001-of-00002.safetensors` (`1,997,108,320` B): `4461aaca4cac9b394ed951697e95a6da3c8952ed5976bbc162c5b085bdc0924a`
  - `model-00002-of-00002.safetensors` (`1,418,232,176` B): `aea734125f5ac9c460847b33836977fe6afa68861b2d687f82bfe7d2b544d699`
  - `model.safetensors.index.json` (`18,805` B): `cc2b212985f5d0469bf926903e4bd7ee81856687baa0c6c6b55ff200d4cdc63f`
  - `tokenizer.json` (`10,115,786` B): `710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c`

## 2. Native BF16 Loading & Actual Execution Path

- Unmodified pinned SGLang (`b2ce7f8811bc40fd0d67000030c22edbbe0f2a37`) loads `Spark2_5ForCausalLM` natively with `--load-format safetensors --dtype bfloat16 --kv-cache-dtype bfloat16 --incremental-streaming-output`, zero missing/unmatched weight warnings.
- Full KV Cache & SWA KV Cache both confirmed `torch.bfloat16`.
- On GB10 (SM121), `Spark2_5MLP` (`GeluAndMul(approximate="tanh")`, `intermediate_size=6144`) dispatches to SGLang's live **JIT CUDA kernel** `python/sglang/kernels/jit/csrc/elementwise/activation.cuh` (`ActivationKind::kGELUTanh`), verified causally by the BF16 positive/negative controls.

## 3. Verified BF16 E0 Baseline, Calibration & Worker Qualification (`spark-c3bc:/var/tmp/ee-l3-bf16-20261126`)

| Artifact | SHA-256 | Result |
|---|---|---|
| `registration.json` | `740179b0b31fd4f0c2324be6ed20997200b88d0c63ecc9dabe563c64c77db6fc` | BF16 E0 registered |
| `baseline/pair.json` (`e0a`/`e0b`) | `24cb242cab62a5ce2648f570166b4839eb13e35564169a7d2cef51f318af9c67` | 2 independent cold builds |
| `freeze/development/requests.json` | `cb30250edc1561abd9ae8f22ce8f3453d6c4e90ae756f2a7f95a0da843f91fa4` | BF16-screened thinking-chat |
| `trajectory/calibration.json` | `d39e907b7541c043a40fa2c43d9af3c4f55f5b63b540bae6bd6591a8a027e1dd` | 3-state trajectory envelope |
| `noise/result.json` (6 pairs E0/E0) | `c76209651d5bc3dea67030769b132fc20969b16bf5b48b738afaa14756465f7e` | `stable=True` |
| `graph/replayed.json` | `fa8b7d62c75f10b88b0cba89fd98eded568a8c991157b7785c4daf748237c0c9` | `valid` + `observed` |
| `split/registration.json` | `76162a1f38151fb638e1ceca515b0e7e5fe57dd69ab95aeed1e23edd2ed34a69` | Verified BF16 overlay |
| `authorization.json` | `43483abadf4eebda4fdd28d7b992a1ee4bcba49fc19a9116f0ad9e163b58ce73` | Metered BF16 Worker lease |
| `qualification-v2/acceptance.json` | `53f7a0068d8123f4326f04d0f82b195c9c55e9b57de63b991c87f4bde350d5d4` | All 9 controls passed (`fault` caught `violation`; `lowprec` flagged FP8 copy; JIT builds ~3s) |

Local controller copies & signed approval plans are staged at:
`C:\Users\rog\AppData\Local\EvolveEngine\l3-bf16-20261126\`

## 4. Open Web Search Fix & Real Pi Verification

- Research boundary stays completely open (`deepseek_open_v1`; no post-E0 time restriction).
- Unified `research_search {query, count?}` (legacy `limit` accepted, conflicts raise `ValueError`), `research_fetch {url}` (returns page 1 directly), `research_read`, and `research_reuse` across `v3_level3_web_research.py`, `v3_level3_rpc.mjs`, `v3_level3_run.py` (`facts.research_guide`), and `launch_live.py` (non-quota engineering judgment prompt).
- End-to-end real Pi + DeepSeek + `WebResearch` verification passed (`PI_EXIT=0`, HTTP 200 on `facts`, `research_search`, `research_fetch`, 0 errors/retries).

## 5. WorkBuddy BF16 E0 GB10 Measurements & Metric Design

Measured on `spark-c3bc` (`BF16` weights + `BF16` KV, `triton`, Radix Cache ON, `context_length=65536`, `max_total_tokens=262144`, `max_running_requests=24`):

- **Memory**: Weights `5.35 GB`; Full KV (262K) `3.50 GB`; SWA KV (210K) `8.40 GB`; **91.85 GB** free VRAM remaining.
- **`c=1` cold**: 16K TTFT `1.56s` / `49.7 tok/s`; 32K TTFT `3.76s` / `44.7 tok/s`; 64K TTFT `9.89s` / `37.1 tok/s`.
- **`c=8` hot prefix**: 8×16K median `40.2 tok/s` (`75% >= 40`, `0% >= 50`, agg `261.5 tok/s`); 8×32K `25.1 tok/s` (262K pool caused slight eviction → formal WorkBuddy profile sets `--max-total-tokens 524288`).
- **`c=20` burst (12K hot)**: median `30.3 tok/s` (`0% >= 40`, agg `580.0 tok/s`, TTFT `211 ms`, P95 ITL `34.6 ms`).
- **Mixed (6 hot 16K + 2 cold 32K)**: hot streams dropped to `19.5 tok/s` (`0% >= 40`), cold TTFT `6.19s` / `25.1 tok/s`.
- **`256K` single-request check**: 250K input succeeded cleanly (TTFT `104.7s`, decode `18.6 tok/s`, `99.09 GB` free).

Implemented without modifying historical Q8_0 / 8K baselines:
- `bf16_workbuddy_64k` launch profile in `evolveengine/v3_level3_deploy.py` (`--context-length 65536 --max-total-tokens 524288 --mem-fraction-static 0.75 --max-running-requests 24`, Radix Cache ON).
- `workbuddy-agent-serving-v1` scoring (`15/20/30/35` weights) in `evolveengine/v3_level3_scoring.py`, separating business SLA attainment (`share_ge_40_tps`, `share_ge_50_tps`, `p95_itl_ms`, `unfinished_count`) from positive-ratio exploration progress, projected in `evolveengine/observatory_data.py`.

## 6. Agent Capability & Reliability Fixes

- `evolveengine/v3_level3_startup.py`: unlocked safe serving knobs (`enable-mixed-chunk`, `disable-overlap-schedule`, `disable-chunked-prefix-cache`, `cuda-graph-max-bs`, `page-size`, `num-continuous-decode-steps`, `swa-full-tokens-ratio`, `schedule-conservativeness`) while keeping precision/model/context/stream flags locked.
- `evolveengine/v3_level3_components.py`: enabled fast (~3s) incremental deployment for `python/sglang/kernels/jit/` and `python/sglang/jit_kernel/` CUDA/C++ headers and subpackage `__init__.py`.

## 7. Ready-to-Launch Admission Commands

Both approval plans passed `launch_live.py` audit-only verification (`{"admitted":true,"paid_calls":0,"worker_started":false}`):

- **BF16 Thinking-Chat (`interactive-tpot-thinking-chat-v4`)**:
  ```powershell
  python experiments/spark26_gguf/level3/launch_live.py `
    --approval "C:\Users\rog\AppData\Local\EvolveEngine\l3-bf16-20261126\approval-bf16.json" `
    --approval-sha256 745367c0e1b7eeab5fe45774432aa5bb438615f67d8a0a850f18d232166a4203 `
    --config experiments/spark26_gguf/level3/objective-exploration-bf16.json `
    --execute <NEW_RUN_DIR> `
    --archive <path\to\sglang.tar.gz> `
    --tokenizer <path\to\deepseek-flash-offline-tokenizer>
  ```
- **BF16 WorkBuddy 64K (`workbuddy-agent-serving-v1`, `15/20/30/35`, `bf16_workbuddy_64k`)**:
  - Remote root: `spark-c3bc:/var/tmp/ee-l3-bf16-wb-20261127` (`registration.json` = `c85a68a20ab4b5969f6742fe49b4d130369b330b4946b9a2165a7f47c10ddc16`)
  - Frozen real-text multi-turn workload (`W1` 2K decode, `W2` 16K cold TTFT, `W3` 8-session 12K hot prefix, `W4` 6-hot 12K + 2-cold 16K): `workbuddy-requests.json` = `e49e1d1528deb4cf6c405fc3d91aa3bd8457a3b0815a70f9ea28f89170de9762`
  - 6-pair E0/E0 noise calibration (`stable=True`, log-envelopes `W1=0.0100, W2=0.0179, W3=0.0281, W4=0.0180`): `noise/result.json` = `5d4b2effe8f32a0fed9fcd65b9dbdde5cd8e2ee34a16ba507e4cfa5a1d20558b`
  - Graph E0 replay (Decode + Prefill CUDA Graphs witnessed): `graph/replayed.json` = `36ab483deb705bc2ea44f77352b99218d3d5841a48028fdeeeba3e574c8b78ec`
  - 9-step BF16 Worker qualification (`fault` caught `violation`, `lowprec` flagged FP8 copy, `positive-measure` observed with full per-scene SLA fields): `qualification/acceptance.json` = `ed130cd4cc81af3c0d71b776e1c4a5539fb94524e7ddcf2386a64acde7eb9031`
  ```powershell
  python experiments/spark26_gguf/level3/launch_live.py `
    --approval "C:\Users\rog\AppData\Local\EvolveEngine\l3-bf16-wb-20261127\approval-bf16-workbuddy.json" `
    --approval-sha256 844dc2d73f341d95f6d5833f2cf0dc2a690cec4c35e1749e8c7296aa250ac292 `
    --config experiments/spark26_gguf/level3/objective-exploration-bf16-workbuddy.json `
    --execute <NEW_RUN_DIR> `
    --archive <path\to\sglang.tar.gz> `
    --tokenizer <path\to\deepseek-flash-offline-tokenizer>
  ```
- **BF16 WorkBuddy 64K v2 (`workbuddy-agent-serving-v2`, `10/15/25/15/15/20`, `bf16_workbuddy_64k_v2`) — RECOMMENDED**:
  - **Scored scenes**: `W1` 10% (2K single decode), `W2` 15% (16K cold TTFT), `W3` 25% (8-session 12K hot P25 steady decode, 4 timed waves, `cached_tokens_min=11,770`), `W4_hot` 15% (6-hot+2-cold mixed hot P25 decode, cold injected at `+700ms`, 4 timed waves), `W4_cold` 15% (mixed cold TTFT), `W5` 20% (20-session 12K burst hot P20 steady decode).
  - **Non-scored mandatory guardrails**: `w4_cold_e2e` + `unfinished_count=0`, `g64k_ttft` & `g64k_decode` (64K cold single-request), `p95_itl` (relaxed floor `0.80`).
  - **16K & 64K long-context logprob quality gate**: E0a vs E0b calibrated (`e0e0_max_tv_upper=4.86e-7`, `e0e0_max_regret=0.0`, limits `0.02 / 0.02`); `anchor-long-ctx.json` = `585d2af297119644d8d4ccea4a380c70ea844664622cf781ce53ecaba65902f6`.
  - Remote root: `spark-c3bc:/var/tmp/ee-l3-bf16-wbv2-20261127` (`registration.json` = `1d501d6b3259dc585c32f46af8d3861c4017fef91929b6a3f7a689553d2fd772`)
  - Frozen workload (`W1–W5 + G64K`): `workbuddy-v2-requests.json` = `d67828213e4032c3507bec6141716f5d1b9aeb7215306119e0ced0d202baef76`
  - 6-pair E0/E0 noise calibration (`stable=True`, scene log-envelopes `W1=0.0100, W2=0.0100, W3=0.0100, W4_cold=0.0100, W4_hot=0.0127, W5=0.0100`, guardrail log-envelopes `w4_cold_e2e=0.0100, g64k_ttft=0.0100, g64k_decode=0.0158, p95_itl=0.0209`, calibrated precision targets all `1.015`): `noise/result.json` = `d01287f589737b99f909e3a98cd75acd37e989437124993144929edb959ad306`
  - Graph E0 replay (Decode + Prefill CUDA Graphs witnessed): `graph/replayed.json` = `2d13f0eabdabdd07ee61fa334067bdcd2e94ce161c78352a649be30a963be004`
  - 13-step BF16 Worker qualification (`fault` caught `violation`, `lowprec` flagged FP8 copy, `longctx_pos` passed 6/6, `longctx_fault` caught exclusively by `long_context_logprob_16k_64k`, `positive-measure` verified with real public-receipt `guardrails.G64K` + `compare_dev` check): `qualification/acceptance.json` = `7330590055de52a3d8443144e8ae24fc00274beb3a7f0d7e381cb2e91c9af7e8`
  - Audit-only check passed (`{"admitted":true,"paid_calls":0,"worker_started":false}`):
  ```powershell
  python experiments/spark26_gguf/level3/launch_live.py `
    --approval "C:\Users\rog\AppData\Local\EvolveEngine\l3-bf16-wbv2-20261127\approval-bf16-workbuddy-v2.json" `
    --approval-sha256 932e10e63cbec580f21479c906fed6efd5de8612aef1f5b4443871e7a6d2d2c6 `
    --config experiments/spark26_gguf/level3/objective-exploration-bf16-workbuddy-v2.json `
    --execute <NEW_RUN_DIR> `
    --archive <path\to\sglang.tar.gz> `
    --tokenizer <path\to\deepseek-flash-offline-tokenizer>
  ```
