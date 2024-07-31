namespace NetML.ML2;

public static partial class TensorExtensions {
    public static Tensor<float> layer_norm(this Tensor<float> x, Tensor<float> gamma, Tensor<float> beta) {
        layer_norm(x, x, gamma, beta);
        return x;
    }

    public static void layer_norm(this Tensor<float> target, Tensor<float> x, Tensor<float> gamma, Tensor<float> beta) {
        const float epsilon = 1e-5f;

        if(target.linear_length != x.linear_length)
            throw new ArgumentException("target and x must be the same length");
        if(gamma.linear_length != (ulong)x.shape[1])
            throw new ArgumentException("gamma is wrong length");
        if(beta.linear_length != (ulong)x.shape[1])
            throw new ArgumentException("beta is wrong length");

        for (var c = 0; c < target.shape[0]; c++) {
            var source_span = x.as_readonly_span([c]);
            var target_span = target.as_span([c]);

            var mean = x.average_and_variance();

            // normalize, scale and shift
            for (var i = 0; i < source_span.Length; i++) {
                target_span[i] = gamma[i] * ((source_span[i] - mean.average) / float.Sqrt(mean.variance + epsilon))
                                 + beta[i];
            }
        }
    }
}