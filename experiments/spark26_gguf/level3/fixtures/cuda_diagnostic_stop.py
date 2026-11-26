# Trusted manual-stop fixture. Keep one CUDA allocation live after the marker.
import time
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.ones(4096, device='cuda')
torch.cuda.synchronize()
print('cuda-stop-fixture-ready', flush=True)
time.sleep(100)
print('unexpected-stop-fixture-completion', flush=True)
