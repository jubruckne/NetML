using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct ReLU<T>: IUnaryOperator<T>, IUnaryStreamOperator<T>
        where T: unmanaged, IFloatingPointIeee754<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public ReLU(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name = name ?? $"{target.name} = relu({source.target.name})";
        }

        public void execute() {
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = apply(evaluated_source[i]);
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(ReLU<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }

        public static T apply(T x)
            => T.Max(T.Zero, x);

        public static Vector128<T> apply(Vector128<T> x)
            => Vector128.Max(Vector128<T>.Zero, x);
    }
}