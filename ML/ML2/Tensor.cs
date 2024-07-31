using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Text;

namespace NetML.ML2;

public unsafe class Tensor<T>
    where T: unmanaged, INumber<T> {

    public string name { get; }

    //ReadOnlySpan<int> ITensor<T, Tensor<T>>.shape => shape;
    //ReadOnlySpan<int> ITensor<T, Tensor<T>>.strides => strides;

    public int[] shape { get; set; }
    public int[] strides { get; set; }
    public ulong linear_length { get; }
    public int rank => shape.Length;

    private readonly T* data;

    private Tensor(string name, T* data, ulong linear_length, int[] shape, int[] strides) {
        this.name          = name;
        this.shape         = shape;
        this.strides       = strides;
        this.linear_length = linear_length;
        this.data          = data;
    }

    public static Tensor<T> create(string name, T* data, ulong linear_length, int[] shape, int[] strides) {
        return new(name, data, linear_length, shape, strides);
    }

    public static Tensor<T> create(string name, T* data, ulong linear_length, ReadOnlySpan<int> shape) {
        var x = TensorExtensions.calculate_strides(shape);

        if (linear_length != x.linear_length)
            throw new Exception($"Tensor {name} has wrong linear length. Calculated {x.linear_length}, but given {linear_length}!");

        return new(name, data, linear_length, x.shape, x.strides);
    }

    public static Tensor<T> create(string name,
                                   T* data,
                                   ulong linear_length,
                                   ReadOnlySpan<int> shape,
                                   ReadOnlySpan<int> strides
    ) {
        return new(name, data, linear_length, shape.ToArray(), strides.ToArray());
    }

    public static Tensor<T> allocate_empty() => new("empty", null, 0, null!, null!);

    public T this[params ReadOnlySpan<int> indices] {
        get {
            var index = calculate_index(indices);
            return data[index];
        }
        set {
            var index = calculate_index(indices);
            data[index] = value;
        }
    }

    public Tensor<T> slice(ReadOnlySpan<int> indices) {
        var index = calculate_column_index_and_length(indices);
        //Console.WriteLine($"slicing {name} [{shape.AsSpan(1).join()}], length: {index.length}");
        return Tensor<T>.create(name, data + index.linear_index, (ulong)index.length, shape.AsSpan(1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> as_span() => new(data, checked((int)linear_length));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> as_span(ulong start) => new(data + start, checked((int)(linear_length - start)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> as_span(ulong start, int length) => new(data + start, int.Min(length, checked((int)(linear_length - start))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> as_span(ulong start, ulong length) => new(data + start, int.Min((int)length, checked((int)(linear_length - start))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> as_span(ReadOnlySpan<int> indices) {
        var index = calculate_column_index_and_length(indices);
        return new(data + index.linear_index, index.length);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> as_readonly_span() => new(data, checked((int)linear_length));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> as_readonly_span(ulong start) => new(data + start, checked((int)(linear_length - start)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> as_readonly_span(ulong start, int length) => new(data + start, int.Min(length, checked((int)(linear_length - start))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> as_readonly_span(ulong start, ulong length) => new(data + start, int.Min((int)length, checked((int)(linear_length - start))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> as_readonly_span(ReadOnlySpan<int> indices) {
        var index = calculate_column_index_and_length(indices);
        return new(data + index.linear_index, index.length);
    }

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

    private (ulong linear_index, int length) calculate_column_index_and_length(ReadOnlySpan<int> indices) {
        if (indices.Length >= rank)
            throw new ArgumentException($"Incorrect number of indices. Expected <{rank} but got {indices.Length}");

        ulong index = 0;
        var i = 0;
        for (; i < indices.Length; i++) {
            index += (ulong)(indices[i] * strides[i]);
        }

        var length = 1;
        for (; i < shape.Length; i++) {
            length *= shape[i];
        }

        return (index, length);
    }

    public string friendly_shape => $"[{shape.join()}]";
    public string friendly_name => $"{name}[{shape.join()}]";

    public void print(int max_values = 8) {
        var sb = new StringBuilder();
        sb.AppendLine($"Tensor: {name}[" + string.Join(", ", shape) + "]");
        sb.AppendLine("Values: ");
        print_values(sb, new int[shape.Length], 0);
        Console.WriteLine(sb.ToString());
    }

    private void print_values(StringBuilder sb, int[] indices, int dimension, int max_values = 8) {
        if (dimension == shape.Length) {
            sb.Append($"{this[indices]:N4} ");
        } else {
            sb.Append("[");
            if (dimension == 0) sb.Append("\n ");

            for (var i = 0; i < int.Clamp(shape[dimension], 0, max_values); i++) {
                indices[dimension] = i;
                print_values(sb, indices, dimension + 1, max_values);
            }

            sb.Remove(sb.Length - 1, 1);
            sb.AppendLine("]");
            if (dimension > 0) sb.Append(" ");
            if (dimension == 0) sb.AppendLine();
        }
    }

    public override string ToString() => name;

    public bool has_same_shape(ReadOnlySpan<int> other_shape)
        => other_shape.SequenceEqual(shape);

    public bool has_same_shape(IEnumerable<int> other_shape)
        => other_shape.SequenceEqual(shape);

    public Tensor<T> clone(string? new_name = default)
        => create(new_name ?? name, data, linear_length, shape.ToArray(), strides.ToArray());
}