using System.Numerics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct Sum<T>: IUnaryOperator<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source { get; }
        public ITensor<T> target { get; }

        public Sum(ITensorOperand<T> source,
                   ITensor<T> target,
                   string? name = default
        ) {
            this.source = source;
            this.target = target;
            this.name   = name ?? $"{target.name} = sum({source.name})";
        }

        public void execute() {
            var evaluated_source = source.evaluate();
            var sum = T.Zero;

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                sum += evaluated_source[i];
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(Sum<T>)} (Name: {name}, Source1: {source.name}, Target: {target.name})";
        }
    }
}