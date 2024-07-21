using System.Numerics;
namespace NetML.ML;

public class ComputationGraph<T> where T: unmanaged, INumber<T> {
    private readonly List<ITensorOperator<T>> operations;

    public ComputationGraph() {
        operations = [];
    }

    private ComputationGraph(List<ITensorOperator<T>> operations) {
        this.operations = operations;
    }

    public void add(ITensorOperator<T> @operator) {
        operations.Add(@operator);
    }

    public ComputationGraph<T> optimize(IGraphOptimizer<T>[] optimizers) {
        var optimized = new List<ITensorOperator<T>>(operations);

        foreach (var optimizer in optimizers) {
            optimized = optimizer.optimize(operations);
        }

        return new(optimized);
    }

    public void execute() {
        foreach (var operation in operations) {
            operation.evaluate();
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