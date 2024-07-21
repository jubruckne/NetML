using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public static partial class Operator {
    public readonly struct RandomUniform<T>:
        IUnaryOperator<T>,
        IUnaryStreamOperator<T>,
        IInitializer<T>
        where T: unmanaged, INumber<T> {
        private const float min = -0.5f;
        private const float max = 0.5f;

        public string name { get; }
        public ITensorOperand<T> source { get; }

        public ITensor<T> target { get; }

        public RandomUniform(ITensorOperand<T> source,
                             ITensor<T> target,
                             string? name = default
        ) {
            this.source = source;
            this.target = target;
            this.name = name ?? $"{target.name} = RandomUniform()";
        }

        public void execute() {
            for (var i = 0; i < target.linear_length; ++i) {
                target[i] = apply();
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(Add<T>)} (Name: {name}, Source: {source.name}, Target: {target.name})";
        }

        public static T apply()
            => T.CreateChecked(Random.Shared.NextDouble() * (max - min) + min);

        public static T apply(T source)
            => apply();

        public static Vector128<T> apply(Vector128<T> arg1)
            => Vector128.Create(
                                [
                                    apply(),
                                    apply(),
                                    apply(),
                                    apply()
                                ]
                               );
    }
}