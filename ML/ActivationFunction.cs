using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NetML.ML;

[SkipLocalsInit]
public static class ActivationFunctions {
    public static unsafe void sigmoid(Vector vector) {
        var ptr = vector.get_pointer();
        for (var i = 0; i < vector.length; i += 4) {
            var v = Vector128.LoadAligned(ptr + i);

            v = Vector128<float>.One
                          / (Vector128<float>.One
                             + Vector128.Exp(-v));

            v.StoreAligned(ptr + i);
        }
    }

    public static unsafe void sigmoid(Vector source, Vector target) {
        var src_ptr = source.get_pointer();
        var tgt_ptr = target.get_pointer();

        for (var i = 0; i <= source.length - 4; i += 4) {
            var v = Vector128.LoadAligned(src_ptr + i);

            v = Vector128<float>.One
                / (Vector128<float>.One
                   + Vector128.Exp(-v));

            v.StoreAligned(tgt_ptr + i);
        }
    }

    public static unsafe void sigmoid_derivative(Vector source, Vector target) {
        var src_ptr = source.get_pointer();
        var tgt_ptr = target.get_pointer();

        for (var i = 0; i < source.length; i += 4) {
            var v = Vector128.LoadAligned(src_ptr + i);
            v *= Vector128<float>.One - v;
            v.StoreAligned(tgt_ptr + i);
        }
    }
}