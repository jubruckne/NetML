using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct OuterProduct<T>: IBinaryOperation<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source1 { get; }
        public ITensorOperand<T> source2 { get; }
        public ITensor<T> target { get; }

        public OuterProduct(ITensorOperand<T> source1,
                                     ITensorOperand<T> source2,
                                     ITensor<T> target,
                                     string? name = default
        ) {
            this.source1 = source1;
            this.source2 = source2;
            this.target  = target;
            this.name    = name ?? $"{target.name} = {source1.target.name} \u2297 {source2.target.name}";
        }

        public void execute() {
            var evaluated_source_1 = source1.evaluate();
            var evaluated_source_2 = source2.evaluate();

            var dim1 = evaluated_source_1.shape[0];
            var dim2 = evaluated_source_2.shape[0];

            if (target.shape.Length != 2 || target.shape[0] != dim1 || target.shape[1] != dim2) {
                throw new InvalidOperationException(
                                                    "The shape of the target tensor does not match the expected result shape."
                                                   );
            }

            for (var i = 0; i < dim1; i++) {
                for (var j = 0; j < dim2; j++) {
                    target[i, j] = evaluated_source_1[i] * evaluated_source_2[j];
                }
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return
                $"{nameof(OuterProduct<T>)} (Name: {name}, Source1: {source1.target.name}, Source2: {source2.target.name}, Target: {target.name})";
        }
    }
}