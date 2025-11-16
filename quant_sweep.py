from typing import List
import subprocess
from itertools import product
# variables from examples/llama3.py
AVAILABLE_MODELS    = [ None ]
AVAILABLE_SIZES     = [("--size", _) for _ in ["1B", "8B", "70B", "405B"]]
# --shard is skipped
# --temperature is skipped
AVAILABLE_QUANTS    = [("--quantize", _) for _ in ["int8", "nf4", "float16", "fp8"]]


# variables to sweep over
SSEEDS  = [("--seed", _) for _ in [42]]
SSIZES  = [("--size", _) for _ in ["1B"]]
SQUANTS = [("--quantize", _) for _ in ["int8", "nf4", "float16", "fp8"]]

SVARS   = [SSEEDS, SSIZES, SQUANTS]

def whoami():
  import platform
  import getpass
  import socket
  from tinygrad import Device

  print(f"OS: {platform.system()} {platform.release()}")
  username = getpass.getuser()
  hostname = socket.gethostname()
  print(f"{username}@{hostname}")
  dev = Device.default
  print(f"Device: {dev}")
  if hasattr(dev, 'iface') and hasattr(dev.iface, 'vram_size'):
    print(f"VRAM: {dev.iface.vram_size / (1024**3):.1f} GB")

# 1. precheck that variables are valid
def is_subset(a: List, b: List) -> bool: 
  _a = [_[1] for _ in a]; _b = [_[1] for _ in b]
  return set(_a) <= set(_b)

assert is_subset(SSIZES,    AVAILABLE_SIZES)
assert is_subset(SQUANTS,   AVAILABLE_QUANTS)

# 2. generate benchmark commands (for subprocess)
configs = list(product(*SVARS))
print(configs)
print(configs[0])
# 3. generate corresponding filename for raw output
# 4. pretty print for dry run
# 5. actually run, and save output to file
# 6. also save device info
