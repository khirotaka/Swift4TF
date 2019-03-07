//
//  data.swift
//  S4TF_1
//
// Reference: https://github.com/tensorflow/swift-models/blob/stable/MNIST/MNIST.swift
//

import Python
import TensorFlow

func readFile(_ filaname: String) -> [UInt8] {
    let d = Python.open(filaname, "rb").read()
    return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
}

func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>, labels: Tensor<Int32>) {
    let images = readFile(imagesFile).dropFirst(16).map { Float($0) }
    let labels = readFile(labelsFile).dropFirst(8).map { Int32($0) }
    let rowCount = Int32(labels.count)
    let columnCount = Int32(images.count) / rowCount
    
    return (
        images: Tensor(shape: [rowCount, columnCount], scalars: images) / 255,
        labels: Tensor(labels)
    )
}

func accuracy(pred: Tensor<Int32>, label: Tensor<Int32>) -> Float {
    return Tensor<Float>(pred .== label).mean().scalarized()
}
