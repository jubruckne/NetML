using System.Numerics;

namespace NetML.ML2;

public static unsafe partial class TensorExtensions {
    public static (int[] shape, int[] strides, ulong linear_length) calculate_strides(ReadOnlySpan<int> shape) {
        var strides = new int[shape.Length];
        strides[^1] = 1;
        ulong linear_length = 1;

        for (var i = 0; i < shape.Length; i++) {
            linear_length *= (ulong)shape[i];
        }

        for (var i = shape.Length - 2; i >= 0; i--) {
            strides[i]    =  strides[i + 1] * shape[i + 1];
        }

        return (shape.ToArray(), strides, linear_length);
    }

    public static int calculate_index(ReadOnlySpan<int> strides, int idx0, int idx1)
        => idx0 * strides[0] + idx1 * strides[1];

    public static Tensor<T> reshape<T, TSelf>(this TSelf tensor, ReadOnlySpan<int> new_shape)
        where T: unmanaged, INumber<T>
        where TSelf: ITensor<T, TSelf> {

        var new_config = calculate_strides(new_shape);

        if (new_config.linear_length != tensor.linear_length) {
            throw new Exception("Tensor view linear length mismatch");
        }

        return Tensor<T>.create(
                                $"{tensor.name}_reshaped",
                                tensor.data_ptr,
                                tensor.linear_length,
                                new_config.shape,
                                new_config.strides
                               );
    }

    public static Tensor<T> permute<T, TSelf>(this TSelf tensor, ReadOnlySpan<int> permutation)
        where T: unmanaged, INumber<T>
        where TSelf: ITensor<T, TSelf> {

        if (permutation.Length == 0) {
            // Default behavior for 2D tensors when no permutation is specified
            if (tensor.rank == 2) {
                permutation = [1, 0];
            } else {
                throw new InvalidOperationException("Transpose with no parameters operation is only supported for 2D tensors.");
            }
        } else {
            if (permutation.Length != tensor.rank) {
                throw new ArgumentException("Permutation must have the same length as the number of dimensions");
            }
        }

        var new_shape   = new int[tensor.shape.Length];
        var new_strides = new int[tensor.strides.Length];
        ulong new_linear_length = 1;

        for (var i = 0; i < tensor.rank; i++) {
            new_shape[i]   = tensor.shape[permutation[i]];
            new_strides[i] = tensor.strides[permutation[i]];
            new_linear_length *= (ulong)new_shape[i];
        }

        if (new_linear_length != tensor.linear_length) {
            throw new Exception("Tensor view linear length mismatch");
        }

        return Tensor<T>.create(
                   $"{tensor.name}_transposed",
                   tensor.data_ptr,
                   tensor.linear_length,
                   new_shape,
                   new_strides
                  );
    }

    public static Tensor<T> broadcast_to<T, TSelf>(this TSelf tensor, ReadOnlySpan<int> new_shape)
        where T: unmanaged, INumber<T>
        where TSelf: ITensor<T, TSelf> {
        if (new_shape.Length < tensor.rank) {
            throw new ArgumentException("New shape cannot have fewer dimensions than the current shape");
        }

        var new_strides = new int[new_shape.Length];

        // Calculate new strides and linear length
        for (var i = new_shape.Length - 1; i >= 0; i--) {
            if (i >= new_shape.Length - tensor.rank) {
                var j = i - (new_shape.Length - tensor.rank);
                if (tensor.shape[j] != 1 && tensor.shape[j] != new_shape[i]) {
                    throw new ArgumentException($"Incompatible dimensions for broadcasting at axis {j}: {tensor.shape[j]} vs {new_shape[i]}");
                }
                new_strides[i] = (tensor.shape[j] == 1) ? 0 : tensor.strides[j];
            } else {
                new_strides[i] = 0;  // New dimensions have stride 0
            }
        }

        return Tensor<T>.create(
                                tensor.name + "_broadcast",
                                tensor.data_ptr,
                                tensor.linear_length,
                                new_shape,
                                new_strides
                               );
    }

    public static void Deconstruct<T>(this T[] array, out T item1, out T item2, out T item3) {
        item1 = default!;
        item2 = default!;
        item3 = default!;

        if(array.Length != 3) throw new ArgumentException("Array length mismatch. Expected 3 elements!");

        if (array.Length > 2) {
            item3 = array[2];
        }

        if (array.Length > 1) {
            item2 = array[1];
        }

        if (array.Length > 0) {
            item1 = array[0];
        }
    }
}