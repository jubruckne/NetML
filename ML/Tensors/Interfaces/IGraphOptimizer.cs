using System.Numerics;

namespace NetML.ML;

public interface IGraphOptimizer<T> where T: unmanaged, INumber<T> {
    List<ITensorOperation<T>> optimize(List<ITensorOperation<T>> operations);
}