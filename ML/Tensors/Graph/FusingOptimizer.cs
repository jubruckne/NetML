using System.Numerics;
using System.Runtime.Intrinsics.Arm;
using NetML.ML;

namespace NetML;

public class FusingOptimizer<T>: IGraphOptimizer<T>
    where T: unmanaged, INumber<T> {

    public List<ITensorOperator<T>> optimize(List<ITensorOperator<T>> operations) {
        List<ITensorOperator<T>> result = [];

        var left = operations[0];

        for (var i = 1; i < operations.Count; i++) {
            var right = operations[i + 1];
            //var xxxx = AdvSimd.Arm64.LoadVector128x2()

            if (left is Operator.Multiply<T> mul && right is Operator.Add<T> add) {
                right = new Operator.MultiplyAdd<T>(mul, add);
                i += 1;
            }

            result.Add(right);
            left = right;
        }

        return result;
    }
}