using System.Numerics;

namespace NetML.ML;

public interface ITensor<T>: ITensorOperand<T>
    where T: unmanaged, INumber<T>
{
    new string name { get; }
    int[] shape { get; }
    int linear_length { get; }
    T this[ReadOnlySpan<int> indices] { get; set; }
    T this[params int[] indices] { get; set; }
    Span<T> as_span();
    ReadOnlySpan<T> as_readonly_span();
    void clear();
}