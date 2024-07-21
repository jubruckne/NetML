using System.Numerics;

namespace NetML.ML;

public interface IGraphOptimizer<T> where T: unmanaged, INumber<T> {
    List<ITensorOperator<T>> optimize(List<ITensorOperator<T>> operations);
}