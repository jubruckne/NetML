using System.Numerics;

namespace NetML.ML;

public interface ITensorOperand<T> where T: unmanaged, INumber<T> {
    string name { get; }
    ITensor<T> target { get; }
    ITensor<T> evaluate();
}

public interface IUnaryOperation<T>: ITensorOperand<T> where T: unmanaged, INumber<T> {
    ITensorOperand<T> source { get; }
}

public interface IBinaryOperation<T>: ITensorOperand<T> where T: unmanaged, INumber<T> {
    ITensorOperand<T> source1 { get; }
    ITensorOperand<T> source2 { get; }
}

public interface ITernaryOperation<T>: ITensorOperand<T> where T: unmanaged, INumber<T> {
    ITensorOperand<T> source1 { get; }
    ITensorOperand<T> source2 { get; }
    ITensorOperand<T> source3 { get; }
}