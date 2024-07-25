using System.Numerics;
using System.Text;

namespace NetML.ML2;

public unsafe class Tensor<T>: ITensor<T, Tensor<T>>
    where T: unmanaged, INumber<T> {

    public string name { get; }

    ReadOnlySpan<int> ITensor<T, Tensor<T>>.shape => shape;
    ReadOnlySpan<int> ITensor<T, Tensor<T>>.strides => strides;

    public int[] shape { get; }
    public int linear_length { get; }
    public int rank => shape.Length;

    private T* data;
    public readonly int[] strides;

    private Tensor(string name, T* data, int linear_length, int[] shape, int[] strides) {
        this.name          = name;
        this.shape         = shape;
        this.strides       = strides;
        this.linear_length = linear_length;
        this.data          = data;
    }

    public static Tensor<T> create(string name, T* data, int linear_length, int[] shape, int[] strides) {
        return new(name, data, linear_length, shape, strides);
    }

    public static Tensor<T> create(string name, T* data, int linear_length, int[] shape) {
        var x = TensorExtensions.calculate_strides(shape);
        return new(name, data, linear_length, shape, x.strides);
    }

    public static Tensor<T> create(string name, T* data, int linear_length, ReadOnlySpan<int> shape, ReadOnlySpan<int> strides) {
        return new(name, data, linear_length, shape.ToArray(), strides.ToArray());
    }

    public static Tensor<T> allocate_empty() => new("empty", null, 0, null!, null!);

    public T this[ReadOnlySpan<int> indices] {
        get {
            var index = calculate_index(indices);
            return data[index];
        }
        set {
            var index = calculate_index(indices);
            data[index] = value;
        }
    }

    public T this[params int[] indices] {
        get => this[new ReadOnlySpan<int>(indices)];
        set => this[new ReadOnlySpan<int>(indices)] = value;
    }

    public Span<T> as_span() => new(data, linear_length);
    public ReadOnlySpan<T> as_readonly_span() => new(data, linear_length);
    public T* data_ptr => data;
    public bool is_continuous => strides[^1] == 1;

    public void clear() => as_span().Clear();

    private int calculate_index(ReadOnlySpan<int> indices) {
        if (indices.Length != shape.Length)
            throw new ArgumentException("Incorrect number of indices");

        var index = 0;
        for (var i = 0; i < indices.Length; i++) {
            index += indices[i] * strides[i];
        }

        return index;
    }

    public void print() {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine($"Tensor: {name}[" + string.Join(", ", shape) + "]");
        sb.AppendLine("Values: ");
        print_values(sb, new int[shape.Length], 0);
        Console.WriteLine(sb.ToString());
    }

    private void print_values(StringBuilder sb, int[] indices, int dimension) {
        if (dimension == shape.Length) {
            sb.Append($"{this[indices]:N4} ");
        } else {
            sb.Append("[");
            if (dimension == 0) sb.Append("\n ");

            for (var i = 0; i < int.Clamp(shape[dimension], 0, 5); i++) {
                indices[dimension] = i;
                print_values(sb, indices, dimension + 1);
            }

            sb.Remove(sb.Length - 1, 1);
            sb.AppendLine("]");
            if (dimension > 0) sb.Append(" ");
            if (dimension == 0) sb.AppendLine();
        }
    }

    public override string ToString() => name;
}