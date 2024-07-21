using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct MultiplyAdd<T>: ITernaryOperator<T>, ITernaryStreamOperator<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source1 { get; }
        public ITensorOperand<T> source2 { get; }
        public ITensorOperand<T> source3 { get; }   // addend
        public ITensor<T> target { get; }

        public MultiplyAdd(ITensorOperand<T> source1,
                           ITensorOperand<T> source2,
                           ITensorOperand<T> source3,
                           ITensor<T> target,
                           string? name = default
        ) {
            this.source1 = source1;
            this.source2 = source2;
            this.source3 = source3;
            this.target  = target;
            this.name    = name ?? $"{target.name} = {source1.name} * {source2.name} + {source3.name}";
        }

        public MultiplyAdd(Multiply<T> mul,
                           Add<T> add,
                           string? name = default
        ): this(mul.source1, mul.source2, add.source2, add.target, name) {}

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
            return $"{nameof(MultiplyAdd<T>)} (Name: {name}, Source1: {source1.name}, Source2: {source2.name}, Source2: {source3.name}, Target: {target.name})";
        }

        public T apply(T arg1, T arg2, T arg3)
            => arg1 * arg2 + arg3;

        public Vector128<T> apply(Vector128<T> arg1, Vector128<T> arg2, Vector128<T> arg3)
            => arg1 + arg2 + arg3;
    }
}