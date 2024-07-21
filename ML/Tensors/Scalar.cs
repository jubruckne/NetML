using System.Numerics;

namespace NetML.ML;

public sealed class Scalar<T>: ITensor<T> where T: unmanaged, INumber<T> {
    public string name { get; }
    public int[] shape { get; }
    public T value { get => scalar_value; set => scalar_value = value; }

    private T scalar_value;

    int ITensor<T>.linear_length => 1;

    public Scalar(T value, string? name = default) {
        this.name = name ?? value.ToString() ?? typeof(T).Name;
        this.shape = [];
        this.value = value;
    }

    T ITensor<T>.this[ReadOnlySpan<int> indices] {
        get => value;
        set => this.value = value;
    }

    T ITensor<T>.this[params int[] indices] {
        get => value;
        set => this.value = value;
    }

    Span<T> ITensor<T>.as_span() => new Span<T>(ref scalar_value);
    ReadOnlySpan<T> ITensor<T>.as_readonly_span() => new ReadOnlySpan<T>(ref scalar_value);

    public void clear() => value = T.Zero;

    public static implicit operator Scalar<T>(T value) => new(value);
    public static implicit operator T(Scalar<T> scalar) => scalar.value;
}