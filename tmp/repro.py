from tinygrad import nn, Tensor

c_in = 1
c_out = 32
k_sz = 5

layers = [
  nn.Conv2d(c_in, c_out, k_sz), Tensor.relu,
  nn.Conv2d(c_out, c_out, k_sz), Tensor.relu,
]

# replace random weights with ones
# there's a bug here that ties the two biases together. george suggests UNIQUE const.

for layer in layers:
  if hasattr(layer,'bias'):
    print(layer.bias, hash(layer.bias)) # they have different hashes
    print(layer.bias.numpy())

print(Tensor.realize(*[p.replace(Tensor.ones_like(p).contiguous()) for p in nn.state.get_parameters(layers)]))


for layer in layers:
  if hasattr(layer,'bias'):
    print(layer.bias, hash(layer.bias)) # they have different hashes
    print(layer.bias.numpy())

