using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct Sigmoid<T>: IUnaryOperation<T>
        where T: unmanaged, IFloatingPointIeee754<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public Sigmoid(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name   = name ?? $"{target.name} = sigmoid({source.target.name})";
        }

        public void execute() {
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = sigmoid(evaluated_source[i]);
            }
        }

        private static T sigmoid(T value)
            => T.One / (T.One + T.Exp(value));

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(Sigmoid<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }
    }
}