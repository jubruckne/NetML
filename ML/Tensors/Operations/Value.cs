using System.Numerics;

namespace NetML.ML;

public static partial class Operation {
    public readonly struct Value<T>: ITensorOperand<T> where T: unmanaged, INumber<T> {
        public string name { get; }

        ITensor<T> ITensorOperand<T>.target => source;

        public ITensor<T> source { get; }
        public ITensor<T> evaluate() => source;

        public Value(ITensor<T> source, string? name = default) {
            this.source = source;
            this.name   = name ?? $"value({source.name})";
        }

        public override string ToString()
            => $"{nameof(Value<T>)} (Name: {name}, Source: {source.name})";
    }
}