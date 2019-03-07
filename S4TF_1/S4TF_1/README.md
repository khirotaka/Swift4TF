# Fully Connected Neural Network using Swift for TensorFlow

memo:
- OS macOS Mojave 10.14.3
- S4TF v0.2 Release 0.2 2019-03-02
- Xcode v10.1 (10B61)

## PyTorch / Chainer style

```swift
import Python
import TensorFlow
PythonLibrary.useVersion(3, 6)

struct Network: Layer {
    var fc1 = Dense<Float>(inputSize: 784, outputSize: 128)
    var fc2 = Dense<Float>(inputSize: 128, outputSize: 64)
    var fc3 = Dense<Float>(inputSize: 64, outputSize: 10)

    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        let l1 = relu(fc1.applied(to: input, in: context))
        let l2 = relu(fc2.applied(to: l1, in: context))
        return fc3.applied(to: l2, in: context)
    }
}
```

## Resources
**[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)**

1. Training Data 
    - Data:  train-images.idx3-ubyte
    - Label:  train-labels.idx1-ubyte
     
2. Test Data ... 
    - Data:  t10k-images.idx3-ubyte
    - Label:  t10k-labels.idx1-ubyte
    
