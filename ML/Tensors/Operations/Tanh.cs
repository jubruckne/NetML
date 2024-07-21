using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct Tanh<T>: IUnaryOperator<T>, IUnaryStreamOperator<T>, IActivation<T>
        where T: unmanaged, IFloatingPointIeee754<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public Tanh(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name = name ?? $"{target.name} = tanh({source.target.name})";
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
            return $"{nameof(Tanh<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }

        public static T apply(T x)
            => T.Tanh(x);

        public static Vector128<T> apply(Vector128<T> x)
            => Vector128.Create(
                                [
                                    T.Tan(x[0]),
                                    T.Tan(x[1]),
                                    T.Tan(x[2]),
                                    T.Tan(x[3])
                                ]
                               );

    }
}