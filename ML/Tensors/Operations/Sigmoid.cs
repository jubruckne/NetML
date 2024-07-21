using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public static partial class Operator {
    public class Sigmoid: Sigmoid<float> {
        public Sigmoid(ITensorOperand<float> source, ITensor<float> target, string? name = default):
            base(source, target, name) {}
    }

    public class Sigmoid<T>: IUnaryOperator<T>, IUnaryStreamOperator<T>, IActivation<T>
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
            Console.WriteLine($"Sigmoid: {source.target.name} -> {target.name}");
            var evaluated_source = source.evaluate();

            for (var i = 0; i < target.linear_length; i++) {
                target[i] = apply(evaluated_source[i]);
            }
        }

        public ITensor<T> evaluate() {
            execute();
            return target;
        }

        public override string ToString() {
            return $"{nameof(Sigmoid<T>)} (Name: {name}, Source1: {source.target.name}, Target: {target.name})";
        }

        public static T apply(T value)
            => T.One / (T.One + T.Exp(value));

        public static Vector128<T> apply(Vector128<T> value) {
            if (value is Vector128<float> v) {
                var r = Vector128<float>.One / (Vector128<float>.One + Vector128.Exp(-v));
                return r.As<float, T>();
            }

            throw new ArgumentException($"unsupported type: {nameof(T)}", nameof(T));
        }

        public static T differentiate(T source)
            => T.One / (T.One + T.Exp(-source));

        public static Vector128<T> differentiate(Vector128<T> arg1) {
            if(arg1 is Vector128<float> fl)
                return (Vector128<float>.One / (Vector128<float>.One + Vector128.Exp(-fl))).As<float, T>();

            if (arg1 is Vector128<double> dbl)
                return (Vector128<double>.One / (Vector128<double>.One + Vector128.Exp(-dbl))).As<double, T>();

            throw new ArgumentException($"unsupported type: {nameof(T)}", nameof(T));
        }
    }
}