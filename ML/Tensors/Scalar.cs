using System.Numerics;

namespace NetML.ML;

public sealed class Scalar<T>: ITensor<T> where T: unmanaged, INumber<T> {
    public string name { get; }
    public int[] shape { get; }
    private T value;

    public Scalar(string name = "") {
        this.name  = name;
        this.shape = [];
        this.value = T.Zero;
    }

    public T this[ReadOnlySpan<int> indices] {
        get => value;
        set => this.value = value;
    }

    public T this[params int[] indices] {
        get => value;
        set => this.value = value;
    }

    public void clear() => value = T.Zero;
}