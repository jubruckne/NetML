namespace NetML.ML2;

public static partial class TensorExtensions {
    public static void token_embedding(this Tensor<float> target, Tensor<float> wte, Tensor<int> input) {
        for (var i = 0; i < input.shape[0]; i++) {
            var tok = input[i];
            wte.as_span([tok]).CopyTo(target.as_span([i]));
        }
    }

    public static void positional_encoding(this Tensor<float> target, Tensor<float> wpe) {
        target.add(target, wpe);
    }
}