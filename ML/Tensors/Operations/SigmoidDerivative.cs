using System.Numerics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct SigmoidDerivative<T>: IUnaryOperator<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public SigmoidDerivative(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name   = name ?? $"{target.name} = sigmoid_derivative({source.name})";
        }

        private static T sigmoid_derivative(T value)
            => value * (T.One - value);

        public void execute() {
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = sigmoid_derivative(evaluated_source[i]);
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(SigmoidDerivative<T>)} (Name: {name}, Source1: {source.name}, Target: {target.name})";
        }
    }
}