/*:
 # Swift for TensorFlow - macOS snapspot 2019-02-13
 ## Using Python library on Swift for TensorFlow

 ### import Numpy
*/

import TensorFlow
import Python
PythonLibrary.useVersion(3, 6)      // Specify the version of Python.
                                    // On macOS, It is referred to `/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/`.
let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")
var a = np.linspace(-5, 5)
var b = np.arange(50)            // of couse, Swift 4 TF can import and use matplotlib.pyplot but on my Mac, couldn't use it on Python 3.6... 😞

// plt.plot(a, b)
// plt.show()

print("variable a: \(a)")
print("variable b: \(b)")

print("-------------------------")
/*:
 ## Auto-differentiation
*/
@differentiable(wrt: x)             // wrt ... with reference to
func f(x: Tensor<Float>) -> Tensor<Float> {
    return x * x
}

let x: Tensor<Float> = Tensor<Float>([1.0, 2.0, 3.0, 4.0, 5.0])
let y = f(x: x)
print("forward: \(y)")


let df = gradient(of: f)
let df_y = df(x)
print("backward: \(df_y)")
