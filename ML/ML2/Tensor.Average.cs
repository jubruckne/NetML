using System.Numerics.Tensors;

namespace NetML.ML2;

public static partial class TensorExtensions {
    public static float average(this Tensor<float> x)
        => TensorPrimitives.Sum(x.as_readonly_span()) / x.linear_length;

    public static (float average, float variance) average_and_variance(this Tensor<float> x) {
        var avg = TensorPrimitives.Sum(x.as_readonly_span()) / x.linear_length;
        var span = x.as_readonly_span();
        float variance = 0;
        for (var i = 0; i < span.Length; i++) {
            variance += (span[i] - avg) * (span[i] - avg);
        }
        return (avg, variance);
    }
}