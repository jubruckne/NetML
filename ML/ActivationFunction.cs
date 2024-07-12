using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NetML.ML;

[SkipLocalsInit]
public static class ActivationFunctions {
    public static unsafe void sigmoid(Vector vector) {
        var ptr = vector.data;
        var length = vector.length;

        int i;
        for (i = 0; i < length - 4; i += 4) {
            var v = Vector128.LoadAligned(ptr + i);

            v = Vector128<float>.One
                          / (Vector128<float>.One
                             + Vector128.Exp(-v));

            v.StoreAligned(ptr + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.LoadAligned(ptr + i);

            v = Vector64<float>.One
                / (Vector64<float>.One
                   + Vector64.Exp(-v));

            v.StoreAligned(ptr + i);
        }
    }

    public static unsafe void sigmoid(Vector source, Vector target) {
        var src_ptr = source.data;
        var tgt_ptr = target.data;
        var length = source.length;

        int i;
        for (i = 0; i < length - 4; i += 4) {
            var v = Vector128.LoadAligned(src_ptr + i);

            v = Vector128<float>.One
                / (Vector128<float>.One
                   + Vector128.Exp(-v));

            v.StoreAligned(tgt_ptr + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.LoadAligned(src_ptr + i);

            v = Vector64<float>.One
                / (Vector64<float>.One
                   + Vector64.Exp(-v));

            v.StoreAligned(tgt_ptr + i);
        }
    }

    public static unsafe void sigmoid_derivative(Vector source, Vector target) {
        var src_ptr = source.data;
        var tgt_ptr = target.data;
        var length = source.length;

        int i;
        for (i = 0; i < length - 4; i += 4) {
            var v = Vector128.LoadAligned(src_ptr + i);
            v *= Vector128<float>.One - v;
            v.StoreAligned(tgt_ptr + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.LoadAligned(src_ptr + i);
            v *= Vector64<float>.One - v;
            v.StoreAligned(tgt_ptr + i);
        }
    }
}