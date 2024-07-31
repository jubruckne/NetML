using System.Numerics;

namespace NetML.ML2;

public static partial class TensorExtensions {
    public static Tensor<T> transpose<T>(this Tensor<T> source)
        where T: unmanaged, INumber<T> {
        var target = source.clone($"{source.name}T");
        transpose(target, source);
        return target;
    }

    public static void transpose<T>(this Tensor<T> target, Tensor<T> source)
        where T: unmanaged, INumber<T> {
        if (target.rank != 2 || source.rank != 2) {
            throw new InvalidOperationException("Transpose is only supported for 2D tensors.");
        }

        if (!target.shape.Reverse().SequenceEqual(source.shape)) {
            throw new InvalidOperationException("Transpose is the wrong shape.");
        }

        target.shape = source.shape;
        target.strides = source.strides;
    }
}