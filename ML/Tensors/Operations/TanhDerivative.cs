using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct TanhDerivative<T>: IUnaryOperation<T> where T: unmanaged, IFloatingPointIeee754<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public TanhDerivative(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name = name ?? $"{target.name} = tanh_derivative({source.target.name})";
        }

        private static T tanh_derivative(T x) {
            var tanh_x = T.Tanh(x);
            return T.One - tanh_x * tanh_x;
        }

        public void execute() {
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = tanh_derivative(evaluated_source[i]);
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(TanhDerivative<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }
    }
}