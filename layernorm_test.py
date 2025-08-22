import os, numpy as np
from tinygrad import Tensor, nn, Device

np.random.seed(42)

BACKEND1 = "METAL"
BACKEND2 = "WEBGPU"

# the goal is to write a grouprnorm kenrel / layernorm kernel (which layernorm
# calls into) that never gets invoked on a tensor past a certain size threshold:
# (256,256,256) can get split into (8, 128,128,128), and have layernorm be called 

from einops import rearrange, repeat

class GroupNorm:
  """
  Applies Group Normalization over a mini-batch of inputs.
  - Paper: https://arxiv.org/abs/1803.08494v3
  """
  def __init__(self, num_groups:int, num_channels:int, eps=1e-5, affine=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Tensor|None = Tensor.ones(num_channels) if affine else None
    self.bias: Tensor|None = Tensor.zeros(num_channels) if affine else None
    
  def __call__(self, x:Tensor) -> Tensor:
    # Get spatial dimensions dynamically
    batch_size = x.shape[0]
    channels_per_group = self.num_channels // self.num_groups
    spatial_dims = x.shape[2:]
    
    # Reshape for layernorm to work as group norm
    x = rearrange(x, 'b (g c) ... -> b g (c ...)', 
                  g=self.num_groups, 
                  c=channels_per_group)
    
    # Apply layernorm (subtract mean and divide stddev)
    x = x.layernorm(eps=self.eps)
    
    # Reshape back to original shape
    x = rearrange(x, 'b g (c ...) -> b (g c) ...', 
                  g=self.num_groups, 
                  c=channels_per_group,
                  # Need to specify spatial shape for unpacking
                  **{f'd{i}': spatial_dims[i] for i in range(len(spatial_dims))})
    
    if self.weight is None or self.bias is None: 
      return x
    
    # Elementwise affine on channels
    # Repeat weight and bias across batch and spatial dimensions
    weight = repeat(self.weight, 'c -> b c ...', 
                    b=batch_size,
                    # Expand spatial dimensions
                    **{f'd{i}': spatial_dims[i] for i in range(len(spatial_dims))})
    bias = repeat(self.bias, 'c -> b c ...', 
                  b=batch_size,
                  **{f'd{i}': spatial_dims[i] for i in range(len(spatial_dims))})
    
    return x * weight + bias

def test_layernorm(b,c1,c2,N):
  print("Channels", b,c1,c2, "Dims", N)
  data = np.random.randn(b, c1, N,N,N).astype(np.float32)
  weights = np.random.randn(c1, c2, 3, 3, 3).astype(np.float32)# * 0.1

  for backend in [BACKEND1, BACKEND2]:
    os.environ.pop(BACKEND2, None) if backend == BACKEND1 else os.environ.update({BACKEND2: '1'})
    Device.DEFAULT = backend
    x = Tensor(data)
    conv = nn.Conv2d(c1, c2, kernel_size=(3, 3, 3), padding=1, bias=False)
    conv.weight.assign(Tensor(weights)).realize()
    
    out = x.layernorm().realize().numpy()
    print(f"{backend:6s}: Min={out.min():.3f}, Max={out.max():.3f}, Mean={out.mean():.3f}")
    if backend == BACKEND1: metal_out = out
    else: print(f"Max diff: {np.abs(metal_out - out).max():.3f}")

# by "breaks" i mean sizable diff. in this case > 1.0. typical diff is 5.0 

b,c1,c2,N = [1,30,30,128] # works 
test_layernorm(b,c1,c2,N)

print("sweep over input dimensions") # ==============
print("="*80)
#b,c1,c2,N = [1,30,30,128+32] # works on everything smaller
#test_layernorm(b,c1,c2,N)

#b,c1,c2,N = [1,30,30,128+32] # breaks on everything bigger
#test_layernorm(b,c1,c2,N)

#b,c1,c2,N = [1,30,30,128+36] # breaks on everything bigger
#test_layernorm(b,c1,c2,N)

print("sweep over inner channels") # ==============
print("="*80)
b,c1,c2,N = [1,5,5,128+36] # channels [5,10,15,30] break
test_layernorm(b,c1,c2,N)
