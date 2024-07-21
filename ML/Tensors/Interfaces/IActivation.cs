using System.Numerics;

namespace NetML.ML;

public interface IActivation;

public interface IActivation<T>: IActivation, IUnaryStreamOperator<T>
    where T: unmanaged, INumber<T> {}