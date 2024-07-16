using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct ReLU<T>: IUnaryOperation<T> where T: unmanaged, IFloatingPointIeee754<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public ReLU(ITensorOperand<T> source, ITensor<T> target, string? name = default) {
            this.source = source;
            this.target = target;
            this.name = name ?? $"{target.name} = relu({source.target.name})";
        }

        private static T relu(T x) => T.Max(T.Zero, x);

        public void execute() {
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = relu(evaluated_source[i]);
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(ReLU<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }
    }
}