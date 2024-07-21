using System.Numerics;

namespace NetML.ML;

public interface IInitializer;

public interface IInitializer<T>: IInitializer, INullaryStreamOperator<T>
    where T: unmanaged, INumber<T>;