import torch, os, platform, subprocess, sys

print("Torch:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("Device count:", torch.cuda.device_count())
print("cuDNN available:", torch.backends.cudnn.is_available())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[{i}] {torch.cuda.get_device_name(i)} - CC {torch.cuda.get_device_capability(i)}")
else:
    # Try to read nvidia-smi if present
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        print("\n==== nvidia-smi ====\n", out)
    except Exception as e:
        print("\nNo nvidia-smi:", e)

print("Python:", sys.version.replace("\n"," "))
print("OS:", platform.platform())
