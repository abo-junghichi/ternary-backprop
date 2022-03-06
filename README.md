# Training Binarized Neural Networks with gradients constraind to -1,0,+1

This code demonstrates a method to train Binarized Neural Networks
with distnases form step-threashold of activation funtion,
which normalization turn into gradients constraind to -1,0,+1.
And, it represents each weight with a integer-accumlator
rather than froating point number.
So this method requires only light operatons at training-time,
which is even supported by instruction sets for microcontrollers
such as RISC-V RV32I.

## How to approximate normalized gradients of step-function to -1,0,+1
This code assume, step-function for negative input is approximated
by exponentiation whose base goes to infinite number.
So when gradients are normalized by any method,
gradients become 0 except nearest one by threshold of step-function.
Since infinite numbers are agnostic for addition and multiplication,
the gradients remained non-zero can be treated as whatever.
The method treats them at easiest way as if they were -1 or +1.

## How to test

```
$ make
```

and it downloads
[the MNIST database](http://yann.lecun.com/exdb/mnist/)
then begin learning.

## References
[Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
