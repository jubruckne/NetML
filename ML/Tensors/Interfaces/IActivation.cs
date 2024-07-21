using System.Numerics;
using System.Runtime.Intrinsics;

namespace NetML.ML;

public interface IActivation;

public interface IActivation<T>: IActivation, IUnaryStreamOperator<T>
    where T: unmanaged, INumber<T> {

    static abstract T differentiate(T source);
    static abstract Vector128<T> differentiate(Vector128<T> arg1);
}