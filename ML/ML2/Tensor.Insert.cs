namespace NetML.ML2;

public static unsafe partial class TensorExtensions {
    public static void insert(this Tensor<int> tensor, ReadOnlySpan<int> values)
        => insert(tensor, 0, values);

    public static void insert(this Tensor<int> tensor, ulong linear_index, ReadOnlySpan<int> values) {
        if((ulong)values.Length + linear_index >= tensor.linear_length)
            throw new IndexOutOfRangeException("Tensor is too small");

        var target = tensor.as_span(linear_index);
        values.CopyTo(target);
    }

    public static void insert(this Tensor<float> tensor, ReadOnlySpan<int> values)
        => insert(tensor, 0, values);

    public static void insert(this Tensor<float> tensor, ReadOnlySpan<float> values)
        => insert(tensor, 0, values);

    public static void insert(this Tensor<float> tensor, ulong linear_index, ReadOnlySpan<float> values) {
        if((ulong)values.Length + linear_index >= tensor.linear_length)
            throw new IndexOutOfRangeException("Tensor is too small");

        var target = tensor.as_span(linear_index);
        values.CopyTo(target);
    }

    public static void insert(this Tensor<float> tensor, ulong linear_index, ReadOnlySpan<int> values) {
        var ptr = tensor.data_ptr + linear_index;

        if((ulong)values.Length + linear_index >= tensor.linear_length)
            throw new IndexOutOfRangeException("Tensor is too small");

        for (var i = 0; i < values.Length; i++) {
            ptr[linear_index + (ulong)i] = float.CreateChecked(values[i]);
        }
    }

    public static void insert(this Tensor<float> tensor, ReadOnlySpan<int> indices, ReadOnlySpan<float> values) {
        var target = tensor.as_span(indices);
        if (values.Length != indices.Length) {
            throw new ArgumentException("Indices length mismatch");
        }
        values.CopyTo(target);
    }
}