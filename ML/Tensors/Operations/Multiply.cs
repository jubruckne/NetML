using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct Multiply<T>: IBinaryOperator<T>, IBinaryStreamOperator<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source1 { get; }
        public ITensorOperand<T> source2 { get; }
        public ITensor<T> target { get; }

        public Multiply(ITensorOperand<T> source1,
                        ITensorOperand<T> source2,
                        ITensor<T> target,
                        string? name = default
        ) {
            this.source1 = source1;
            this.source2 = source2;
            this.target  = target;
            this.name    = name ?? $"{target.name} = {source1.name} * {source2.name}";
        }

        public void execute() {
            var evaluated_source_1 = source1.evaluate();
            var evaluated_source_2 = source2.evaluate();

            for (var i = 0; i < target.shape.Aggregate(1, static (a, b) => a * b); i++) {
                target[i] = evaluated_source_1[i] * evaluated_source_2[i];
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(Multiply<T>)} (Name: {name}, Source1: {source1.name}, Source2: {source2.name}, Target: {target.name})";
        }

        public static T apply(T source1, T source2)
            => source1 * source2;

        public static Vector128<T> apply(Vector128<T> source1, Vector128<T> source2)
            => source1 * source2;
    }
}