using System.Numerics;
namespace NetML.ML;

public class ComputationGraph<T> where T: unmanaged, INumber<T> {
    private readonly List<ITensorOperation<T>> operations;

    public ComputationGraph() {
        operations = [];
    }

    private ComputationGraph(List<ITensorOperation<T>> operations) {
        this.operations = operations;
    }

    public void add(ITensorOperation<T> operation) {
        operations.Add(operation);
    }

    public ComputationGraph<T> optimize(IGraphOptimizer<T>[] optimizers) {
        var optimized = new List<ITensorOperation<T>>(operations);

        foreach (var optimizer in optimizers) {
            optimized = optimizer.optimize(operations);
        }

        return new(optimized);
    }

    public void execute() {
        foreach (var operation in operations) {
            operation.execute();
        }
    }

    public void clear() {
        operations.Clear();
    }

    public void print() {
        Console.WriteLine("Computation Graph:");
        foreach (var operation in operations) {
            Console.WriteLine(operation);
        }
    }
}