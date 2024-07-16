using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct MatrixMultiplication<T>: IBinaryOperation<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source1 { get; }
        public ITensorOperand<T> source2 { get; }
        public ITensor<T> target { get; }

        public MatrixMultiplication(ITensorOperand<T> source1,
                                    ITensorOperand<T> source2,
                                    ITensor<T> target,
                                    string? name = default
        ) {
            this.source1 = source1;
            this.source2 = source2;
            this.target  = target;
            this.name    = name ?? $"{target.name} = {source1.name} \u00B7 {source2.name}";
        }

        public void execute() {
            var evaluated_source_1 = source1.evaluate();
            var evaluated_source_2 = source2.evaluate();

            if (evaluated_source_1.shape.Length != 2 || evaluated_source_2.shape.Length != 2) {
                throw new InvalidOperationException("MatrixMultiplicationOperation requires two 2D tensors.");
            }

            var rows1 = evaluated_source_1.shape[0];
            var cols1 = evaluated_source_1.shape[1];
            var rows2 = evaluated_source_2.shape[0];
            var cols2 = evaluated_source_2.shape[1];

            if (cols1 != rows2) {
                throw new InvalidOperationException(
                                                    "The number of columns of the first matrix must match the number of rows of the second matrix."
                                                   );
            }

            if (target.shape.Length != 2 || target.shape[0] != rows1 || target.shape[1] != cols2) {
                throw new InvalidOperationException(
                                                    "The shape of the target tensor does not match the expected result shape."
                                                   );
            }

            for (var i = 0; i < rows1; i++) {
                for (var j = 0; j < cols2; j++) {
                    var sum = default(T);
                    for (var k = 0; k < cols1; k++) {
                        sum += evaluated_source_1[i, k] * evaluated_source_2[k, j];
                    }

                    target[i, j] = sum;
                }
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return
                $"{nameof(MatrixMultiplication<T>)} (Name: {name}, Source1: {source1.name}, Source2: {source2.name}, Target: {target.name})";
        }
    }
}