# Trusted engineering fixture; no weights and no performance verdict.
import json
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.arange(128, dtype=torch.int32, device='cuda')
assert int(x.sum().item()) == 8128
print(json.dumps(dict(cuda_devices=torch.cuda.device_count(), result=8128, non_scoring=True)), flush=True)
