using System.Numerics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct ReLUDerivative<T>: IUnaryOperator<T> where T: unmanaged, IFloatingPointIeee754<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public ReLUDerivative(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name = name ?? $"{target.name} = relu_derivative({source.target.name})";
        }

        private static T relu_derivative(T x) => x > T.Zero ? T.One : T.Zero;

        public void execute() {
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = relu_derivative(evaluated_source[i]);
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(ReLUDerivative<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }
    }
}