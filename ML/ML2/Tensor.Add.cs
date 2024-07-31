using System.Numerics.Tensors;

namespace NetML.ML2;

public static partial class TensorExtensions {
    public static Tensor<float> add(this Tensor<float> target, Tensor<float> other) {
        add(target, target, other);
        return target;
    }

    public static void add(this Tensor<float> target, Tensor<float> left, Tensor<float> right) {
        Console.WriteLine($"{target.friendly_name} = {left.friendly_name} + {right.friendly_name}");
        if (left.rank == right.rank) {
            if (target.linear_length == left.linear_length && left.linear_length != right.linear_length) {
                throw new Exception("Dimensions mismatch");
            }

            TensorPrimitives.Add(left.as_readonly_span(), right.as_readonly_span(), target.as_span());
            return;
        }

        if (left.rank > right.rank && target.rank == left.rank) {
            Console.WriteLine($"broadcasting {right.friendly_name} to {target.friendly_shape}...");

            if(target.linear_length % right.linear_length != 0)
                throw new Exception("Dimensions mismatch");

            for (ulong i = 0; i < target.linear_length / right.linear_length; i++) {
                TensorPrimitives.Add(
                                     left.as_readonly_span(i * right.linear_length, right.linear_length),
                                     right.as_readonly_span(),
                                     target.as_span(i * right.linear_length, right.linear_length)
                                    );
            }

            return;
        }

        throw new ArgumentException($"Tensors are not compatible {left.friendly_name} != {right.friendly_name}");
    }
}