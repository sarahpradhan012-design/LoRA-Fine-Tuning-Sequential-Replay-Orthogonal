import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not download the model or dataset yet.")

name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"GPU: {name}")
print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")
# Verify real device allocation, not only device discovery.
x = torch.ones((1024, 1024), device="cuda", dtype=torch.float16)
y = x @ x
print(f"CUDA allocation/matmul: OK ({y[0, 0].item():.0f})")
del x, y
torch.cuda.empty_cache()
