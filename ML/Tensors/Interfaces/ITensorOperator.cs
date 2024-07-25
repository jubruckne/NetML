using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public interface ITensorOperand<T> where T: unmanaged, INumber<T> {
    string name { get; }
    ITensor<T> target => (ITensor<T>)this;
    ITensor<T> evaluate() => (ITensor<T>)this;
}

public interface ITensorOperator<T>: ITensorOperand<T>
    where T: unmanaged, INumber<T>;

public interface INullaryOperator<T>: ITensorOperator<T>
    where T: unmanaged, INumber<T>;

public interface INullaryStreamOperator<T>: ITensorOperator<T>
    where T: unmanaged, INumber<T> {

    static abstract T apply_scalar();
    static abstract Vector128<T> apply_vector();
}

public interface IUnaryOperator<T>: ITensorOperator<T> where T: unmanaged, INumber<T> {
    ITensorOperand<T> source { get; }
}

public interface IUnaryStreamOperator<T>: ITensorOperator<T> where T: unmanaged, INumber<T> {
    static abstract T apply(T source);
    static abstract Vector128<T> apply(Vector128<T> arg1);
}

public interface IBinaryOperator<T>: ITensorOperator<T> where T: unmanaged, INumber<T> {
    ITensorOperand<T> source1 { get; }
    ITensorOperand<T> source2 { get; }
}

public interface IBinaryStreamOperator<T>: ITensorOperator<T> where T: unmanaged, INumber<T> {
    static abstract T apply(T source1, T source2);
    static abstract Vector128<T> apply(Vector128<T> arg1, Vector128<T> arg2);
}

public interface ITernaryOperator<T>: ITensorOperator<T> where T: unmanaged, INumber<T> {
    ITensorOperand<T> source1 { get; }
    ITensorOperand<T> source2 { get; }
    ITensorOperand<T> source3 { get; }
}

public interface ITernaryStreamOperator<T>: ITensorOperator<T> where T: unmanaged, INumber<T> {
    T apply(T source1, T source2, T source3);
    Vector128<T> apply(Vector128<T> arg1, Vector128<T> arg2, Vector128<T> arg3);
}