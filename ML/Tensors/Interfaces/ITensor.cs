using System.Numerics;

namespace NetML.ML;

public interface ITensor<T> where T: unmanaged, INumber<T> {
    string name { get; }
    int[] shape { get; }
    T this[ReadOnlySpan<int> indices] { get; set; }
    T this[params int[] indices] { get; set; }
    void clear();
}