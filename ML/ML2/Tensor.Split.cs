namespace NetML.ML2;

public static partial class TensorExtensions {
    public static Tensor<float>[] split(this Tensor<float> source, int chunk_size) {
        var names = new string[chunk_size];
        for (var i = 0; i < chunk_size; ++i) {
            names[i] = $"source_{i}";
        }

        return split(source, names);
    }

    public static unsafe Tensor<float>[] split(this Tensor<float> source, params string[] names) {
        var chunks = names.Length;
        var chunk_size = (ulong)(source.shape[^1] / chunks);

        if (source.linear_length % chunk_size != 0) {
            throw new ArgumentException("The given tensor's size must be a multiple of the chunk size");
        }

        Console.WriteLine($"splitting {source.friendly_name} into {chunks} chunks...");

        ReadOnlySpan<int> new_shape = [..source.shape[0..^1], (int)chunk_size];

        var config = TensorExtensions.calculate_strides(new_shape);
        Console.WriteLine($"new_shape: [{config.shape.join()}], strides: [{config.strides.join()}], length: {config.linear_length}");

        var t = new Tensor<float>[chunks];

        for (var i = 0; i < chunks; i++) {
            t[i] = Tensor<float>.create(
                                        names[i],
                                        source.data_ptr + (ulong)i * chunk_size,
                                        chunk_size,
                                        config.shape,
                                        config.strides
                                       );
            t[i].print();
        }

        return t;
    }
}