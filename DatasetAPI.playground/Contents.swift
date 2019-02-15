/*:
 # Swift for TensorFlow Dataset API.
 ## Load Iris Dataset (Keras)
*/

import TensorFlow
import Python
PythonLibrary.useVersion(2, 7)                              // Use Python 2.7
let os = Python.import("os")
let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")                // it works properly with Python 2.7, Doesn't work Python 3.6 on macOS. :(

os.chdir(".keras/datasets")        // required full path. on Playgroud, default work dir is `/`.
os.getcwd()                                                 // check.

// --------------------------------------------------------------------------------------------------------
// Reference: https://github.com/tensorflow/swift-tutorials/blob/master/iris/TutorialDatasetCSVAPI.swift
extension Dataset where Element == TensorPair<Tensor<Float>, Tensor<Int32>> {
    public init(contentsOfCSVFile: String, hasHeader: Bool,
                featureColumns: [Int],
                labelColumns: [Int]) {
        // We can't make `np` a private top-level variable in this file, because
        // this function is @inlinable.
        let np = Python.import("numpy")
        
        let featuresNp = np.loadtxt(contentsOfCSVFile, delimiter: ",",
                                    skiprows: hasHeader ? 1 : 0,
                                    usecols: featureColumns,
                                    dtype: Float.numpyScalarTypes.first!)
        guard let featuresTensor = Tensor<Float>(numpy: featuresNp) else {
            // This should never happen, because we construct numpy in such a
            // way that it should be convertible to tensor.
            fatalError("np.loadtxt result can't be converted to Tensor")
        }
        
        let labelsNp = np.loadtxt(contentsOfCSVFile, delimiter: ",",
                                  skiprows: hasHeader ? 1 : 0,
                                  usecols: labelColumns,
                                  dtype: Int32.numpyScalarTypes.first!)
        guard let labelsTensor = Tensor<Int32>(numpy: labelsNp) else {
            // This should never happen, because we construct numpy in such a
            // way that it should be convertible to tensor.
            fatalError("np.loadtxt result can't be converted to Tensor")
        }
        
        self.init(elements: TensorPair(featuresTensor, labelsTensor))
    }
}

extension Sequence where Element == TensorPair<Tensor<Float>, Tensor<Int32>> {
    var first: TensorPair<Tensor<Float>, Tensor<Int32>>? {
        return first(where: {_ in true})
    }
}
// --------------------------------------------------------------------------------------------------------

/*:
 ## Road Iris Dataset.
 */

let trainDataset: Dataset<TensorPair<Tensor<Float>, Tensor<Int32>>> = Dataset(
    contentsOfCSVFile: "iris_training.csv", hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(64)


let firstTrainExamples = trainDataset.first!
let firstTrainFeatures = firstTrainExamples.first
let firstTrainLabels = firstTrainExamples.second
firstTrainFeatures

let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
let petalLengths = firstTrainFeaturesTransposed[3].scalars
let sepalLengths = firstTrainFeaturesTransposed[0].scalars

plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
plt.xlabel("Petal length")
plt.ylabel("Sepal length")

plt.show()
