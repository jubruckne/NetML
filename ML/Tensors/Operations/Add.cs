using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct Add<T>: IBinaryOperation<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source1 { get; }
        public ITensorOperand<T> source2 { get; }
        public ITensor<T> target { get; }

        public Add(ITensorOperand<T> source1,
                            ITensorOperand<T> source2,
                            ITensor<T> target,
                            string? name = default
        ) {
            this.source1 = source1;
            this.source2 = source2;
            this.target  = target;
            this.name    = name ?? $"{target.name} = {source1.name} + {source2.name}";
        }

        public void execute() {
            var evaluated_source_1 = source1.evaluate();
            var evaluated_source_2 = source2.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = evaluated_source_1[i] + evaluated_source_2[i];
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(Add<T>)} (Name: {name}, Source1: {source1.name}, Source2: {source2.name}, Target: {target.name})";
        }
    }
}