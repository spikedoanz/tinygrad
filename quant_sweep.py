from typing import List
import uuid
import subprocess
from itertools import product, chain

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
  return {
    "platform": platform.system(), "release": platform.release(), "device": Device.default,
    "username": getpass.getuser(), "hostname": socket.gethostname()
  }


# 1. precheck that variables are valid
def is_subset(a: List, b: List) -> bool: 
  _a = [_[1] for _ in a]; _b = [_[1] for _ in b]
  return set(_a) <= set(_b)

assert is_subset(SSIZES,    AVAILABLE_SIZES)
assert is_subset(SQUANTS,   AVAILABLE_QUANTS)

# 2. generate benchmark commands (for subprocess)
configs = list(product(*SVARS))

# 3. generate corresponding filename for raw output
def config_to_filename_and_metadata(config) -> tuple[str, dict[str, str]]:
  whoiam = whoami()
  config_dict = {k: v for tup in config for k, v in [tup]}
  parts = [
    whoiam['hostname'],
    config_dict['--size'],
    config_dict['--quantize'],
    f"seed{config_dict['--seed']}",
    f"uuid{str(uuid.uuid4())[:8]}"
  ]
  filename = '_'.join(parts) + '.json'
  metadata = {
    'config': config_dict,
    'whoami': whoiam,
    'command': command_header + list(chain.from_iterable(config)),
    'uuid': parts[-1]
  }
  return filename, metadata


# 4. pretty print for dry run
num_runs = 1
command_header = ["PYTHONPATH=.", "python", "examples/llama3.py"]
for config in configs[:num_runs]:
  filename, metadata = config_to_filename_and_metadata(config)
  print(filename)

# 5. actually run, and save output to file
# 6. also save device info
