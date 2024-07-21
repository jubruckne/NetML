using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public static partial class Operator {
    public static void apply<TStreamOperator>(ReadOnlySpan<float> source, Span<float> target)
        where TStreamOperator: IUnaryStreamOperator<float> {
        for (var i = 0; i < source.Length; i += 4) {
            var v = Vector128.LoadUnsafe(in source[i]);
            var result = TStreamOperator.apply(v);
            result.StoreUnsafe(ref target[i]);
        }
    }

    public static void apply_inplace<TStreamOperator>(Span<float> target)
        where TStreamOperator: IUnaryStreamOperator<float> {
        for (var i = 0; i < target.Length; i += 4) {
            var v      = Vector128.LoadUnsafe(in target[i]);
            var result = TStreamOperator.apply(v);
            result.StoreUnsafe(ref target[i]);
        }
    }
}