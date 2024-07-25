using System.Numerics;

namespace NetML.ML2;

public unsafe interface ITensor<T, out TSelf>
    where T: unmanaged, INumber<T>
    where TSelf: ITensor<T, TSelf>
{
    string name { get; }
    ReadOnlySpan<int> shape { get; }
    ReadOnlySpan<int> strides { get; }
    int rank { get; }
    int linear_length { get; }
    T this[ReadOnlySpan<int> indices] { get; set; }
    T this[params int[] indices] { get; set; }
    Span<T> as_span();
    ReadOnlySpan<T> as_readonly_span();
    T* data_ptr { get; }

    bool is_continuous { get; }

    void clear();
}