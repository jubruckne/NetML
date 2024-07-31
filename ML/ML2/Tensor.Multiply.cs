using System.Numerics.Tensors;

namespace NetML.ML2;

public static partial class TensorExtensions {
    public static void multiply(this Tensor<float> target, Tensor<float> other)
        => multiply(target, target, other);

    public static void multiply(this Tensor<float> target, Tensor<float> left, Tensor<float> right) {
        if (left.rank == 1 && right.rank == 1) {
            TensorPrimitives.Multiply(left.as_readonly_span(), right.as_readonly_span(), target.as_span());
        }
    }
}