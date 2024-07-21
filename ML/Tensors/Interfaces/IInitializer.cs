using System.Numerics;

namespace NetML.ML;

public interface IInitializer;

public interface IInitializer<T>: IInitializer, IUnaryStreamOperator<T>
    where T: unmanaged, INumber<T>;