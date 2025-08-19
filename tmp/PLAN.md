https://discord.com/channels/1068976834382925865/1069001075828469790/1406306074252017736

-[] reproduce
-[] [TRACE]: trace and understand the code



--------------------------------------------------------------------------------
# TRACE
TRACE questions:
-[] 1. how are conv biases stored in nn.Conv2d?
> it's just .bias, but nn.state.get_parameters is the more idiomatic way?

-[] 2. how can we check if they're unique or not?
> they have different hashes
-> is the binder for assignment shared then?
-> numerically they're different and the hashes are different. what does he mean by they're the same tensor?

  -[] NOOPT=1 VIZ=1 DEBUG=2 python test/test_tiny.py TestTiny.test_mnist_backward
    -[] how does this show that they're the same?
    -[] figure out how to read the tinygrad e_a_b_c_d... notation.

-[] 3. how are they constructed / reassigned?
