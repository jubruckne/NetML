using System.Numerics;

namespace NetML.ML;

public sealed class Tensor<T>: ITensor<T> where T: unmanaged, INumber<T> {
    public string name { get; }
    public int[] shape { get; }
    public int linear_length { get; }

    private readonly T[] array;
    private readonly int[] strides;

    public Tensor(int[] shape, string name = "") {
        this.name = name;
        this.shape = shape;
        this.strides = calculate_strides(shape);
        this.linear_length = this.shape.Aggregate(1, static (a, b) => a * b);
        this.array = new T[linear_length];
    }

    public T this[ReadOnlySpan<int> indices] {
        get {
            var index = calculate_index(indices);
            return array[index];
        } set {
            var index = calculate_index(indices);
            array[index] = value;
        }
    }

    public T this[params int[] indices] {
        get => this[new ReadOnlySpan<int>(indices)];
        set => this[new ReadOnlySpan<int>(indices)] = value;
    }

    public Span<T> as_span() => array;

    public ReadOnlySpan<T> as_readonly_span() => array;

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

    private int[] calculate_strides(ReadOnlySpan<int> shape) {
        var strides = new int[shape.Length];
        strides[shape.Length - 1] = 1;

        for (var i = shape.Length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    public void print() {
        Console.WriteLine($"Tensor: {name}");
        Console.WriteLine("Shape: [" + string.Join(", ", shape) + "]");
        print_values(new int[shape.Length], 0);
    }

    private void print_values(int[] indices, int dimension) {
        if (dimension == shape.Length) {
            Console.Write(this[indices] + " ");
        } else {
            Console.Write("[");
            for (var i = 0; i < shape[dimension]; i++) {
                indices[dimension] = i;
                print_values(indices, dimension + 1);
            }
            Console.Write("]");
            if (dimension > 0) Console.Write(" ");
            if (dimension == 0) Console.WriteLine();
        }
    }
}