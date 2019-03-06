//
//  main.swift
//  S4TF_1 MNIST
//
//  Created by 川島 寛隆 on 2019/03/04.
//  Copyright © 2019 川島 寛隆. All rights reserved.
//
import Foundation
import TensorFlow
import Python
PythonLibrary.useVersion(3, 6)
let np = Python.import("numpy")


let (x_train, y_train) = readMNIST(imagesFile: "Resources/train-images.idx3-ubyte",
                                    labelsFile: "Resources/train-labels.idx1-ubyte")

let (x_test, y_test) = readMNIST(imagesFile: "Resources/t10k-images.idx3-ubyte",
                                 labelsFile: "Resources/t10k-labels.idx1-ubyte")

let trainImage = Dataset<Tensor<Float>>(elements: x_train)
let trainLabel = Dataset<Tensor<Int32>>(elements: y_train)

let testImage = Dataset<Tensor<Float>>(elements: x_test)
let testLabel = Dataset<Tensor<Int32>>(elements: y_test)


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


let numEpochs = 20
let batchSize: Int64 = 64
let optimzier = SGD<Network, Float>(learningRate: 0.02)
let trainContext = Context(learningPhase: LearningPhase.training)
var model = Network()


func train() {
    var epochLoss: Float = 0.0
    var epochAcc: Float = 0.0
    var count: Int = 0
    
    for epoch in stride(from: 0, to: numEpochs, by: 1) {
        for (x, y) in zip(trainImage.batched(batchSize), trainLabel.batched(batchSize)) {
            let (loss, grad) = model.valueWithGradient { (model: Network) -> Tensor<Float> in
                let logit = model.applied(to: x, in: trainContext)
                return softmaxCrossEntropy(logits: logit, labels: y)
            }
            optimzier.update(&model.allDifferentiableVariables, along: grad)
            
            let pred = model.applied(to: x, in: trainContext)
            epochAcc += accuracy(pred: pred.argmax(squeezingAxis: 1), label: y)
            epochLoss += loss.scalarized()
            count += 1
        }
        
        print("Epoch \(epoch+1) / \(numEpochs): Loss: \(epochLoss / Float(count)) | Accuracy: \(epochAcc / Float(count))")
    }
}

func test() {
    var testLoss: Float = 0.0
    var testAcc: Float = 0.0
    var count: Int = 0
    
    for (x, y) in zip(testImage.batched(batchSize), testLabel.batched(batchSize)) {
        let logits = model.inferring(from: x)
        let pred = logits.argmax(squeezingAxis: 1)
        let loss = softmaxCrossEntropy(logits: logits, labels: y)
        
        testLoss += loss.scalarized()
        testAcc += Tensor<Float>(pred .== y).mean().scalarized()
        count += 1
    }
    print("----------------------------------------------------------------------")
    print("Result")
    print("loss: \(testLoss / Float(count)) | Accuracy: \(testAcc / Float(count))")
}

train()
test()
