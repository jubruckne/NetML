using System.Numerics;
using System.Text;

namespace NetML.ML2;

public class Graph: GraphNode {
    public List<GraphNode> nodes { get; }

    protected Graph(Tensor<float> target): base(target) {
        nodes = [];
    }

    protected override void apply(Tensor<float> target) {
        foreach (var node in nodes) {
            node.execute();
        }
    }

    public static Graph with_target(Tensor<float> target) => new Graph(target);

    public override string ToString() {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine($"Graph for {target}");
        foreach (var node in nodes) {
            sb.AppendLine($"  {node}");
        }

        return sb.ToString();
    }

    public Graph add(GraphNode target, GraphNode other) {
        nodes.Add(new Add_float_float_float(target.target, other, target));
        return this;
    }

    public Graph add(GraphNode target, GraphNode left, GraphNode right) {
        nodes.Add(new Add_float_float_float(target.target, left, right));
        return this;
    }

    public Graph multiply(GraphNode target, GraphNode other) {
        nodes.Add(new Multiply_float_float_float(target.target, other, target));
        return this;
    }

    public Graph sigmoid(GraphNode target) {
        nodes.Add(new Sigmoid_float_float(target.target, target));
        return this;
    }

    public Graph sigmoid(GraphNode target, GraphNode source) {
        nodes.Add(new Sigmoid_float_float(target.target, source));
        return this;
    }
}

public static class GraphExtensions {
    public static GraphNode as_node(this Tensor<float> vector)
        => new ValueNode(vector);
}