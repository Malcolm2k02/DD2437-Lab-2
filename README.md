## Part 1 – RBF Networks

### 3.1 Batch Learning

The first experiment approximates:

- `sin(2x)`
- `square(2x)`

using Gaussian RBF units and batch least-squares learning.

The RBF activation is

φ(x) = exp(-||x - μ||² / (2σ²))

The output weights are learned using least squares.

For `sin(2x)`, the following minimum numbers of RBF units were found using σ = 0.5:

| Error threshold | RBF units |
|-----------------|-----------|
| < 0.1           | 8         |
| < 0.01          | 12        |
| < 0.001         | 20        |

For the square wave, applying a threshold to the network output

- output >= 0 → +1
- output < 0 → -1

produced zero classification error with 9 RBF units.

### Answers to the Assignment Questions

How many RBF units are required for sin(2x)?

Error < 0.1: 8 RBF units
Error < 0.01: 12 RBF units
Error < 0.001: 20 RBF units

Why is the square wave more difficult to approximate?

Gaussian RBFs are smooth functions, whereas the square wave contains discontinuous jumps between +1 and -1. The RBF network therefore cannot reproduce the sharp transitions as easily as it can approximate the smooth sine function.

What transformation can reduce the square-wave residual error to zero?

A threshold/sign transformation can be applied to the network output, mapping outputs greater than or equal to zero to +1 and negative outputs to -1.

How many RBF units were required for zero transformed error?

In this experiment, 9 RBF units were sufficient to obtain zero residual error on the test set.

Where could this transformation be useful?

The transformation is useful in binary classification, where the sign or class of the output is more important than its exact numerical magnitude.

### 3.2 Regression with noise
