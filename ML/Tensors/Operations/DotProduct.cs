using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct DotProduct<T>: IBinaryOperation<T> where T: unmanaged, INumber<T> {
        public string name { get; }
        public ITensorOperand<T> source1 { get; }
        public ITensorOperand<T> source2 { get; }
        public ITensor<T> target { get; }

        public DotProduct(ITensorOperand<T> source1,
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

            var shape1 = evaluated_source_1.shape;
            var shape2 = evaluated_source_2.shape;

            if (shape1.Length == 0 || shape2.Length == 0) {
                throw new InvalidOperationException("DotProductOperation requires tensors of at least rank 1.");
            }

            if (shape1[^1] != shape2[0]) {
                throw new InvalidOperationException(
                                                    "The last dimension of the first tensor must match the first dimension of the second tensor."
                                                   );
            }

            var result_rank  = shape1.Length + shape2.Length - 2;
            var result_shape = new int[result_rank];

            Array.Copy(shape1, result_shape, shape1.Length - 1);
            Array.Copy(shape2, 1, result_shape, shape1.Length - 1, shape2.Length - 1);

            if (!result_shape.SequenceEqual(target.shape)) {
                throw new InvalidOperationException(
                                                    "The shape of the target tensor does not match the expected result shape."
                                                   );
            }

            var dim         = shape1[^1];
            var result_size = result_shape.Aggregate(1, static (a, b) => a * b);

            for (var i = 0; i < result_size; i++) {
                var sum           = default(T);
                var resultIndices = get_indices(result_shape, i);

                for (var j = 0; j < dim; j++) {
                    var source_1_indices = new int[shape1.Length];
                    var source_2_indices = new int[shape2.Length];

                    Array.Copy(resultIndices, source_1_indices, shape1.Length - 1);
                    source_1_indices[shape1.Length - 1] = j;

                    source_2_indices[0] = j;
                    Array.Copy(resultIndices, shape1.Length - 1, source_2_indices, 1, shape2.Length - 1);

                    sum += evaluated_source_1[source_1_indices] * evaluated_source_2[source_2_indices];
                }

                target[resultIndices] = sum;
            }
        }

        private int[] get_indices(int[] shape, int flatIndex) {
            var indices = new int[shape.Length];
            for (var i = shape.Length - 1; i >= 0; i--) {
                indices[i] =  flatIndex % shape[i];
                flatIndex  /= shape[i];
            }

            return indices;
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(DotProduct<T>)} (Name: {name}, Source1: {source1.name}, Source2: {source2.name}, Target: {target.name})";
        }
    }
}