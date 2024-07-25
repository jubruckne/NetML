using NetML.ML;

namespace NetML.ML2;

public abstract class GraphNode {
    protected virtual string node_name => "node";
    public Tensor<float> target { get; }
    protected abstract void apply(Tensor<float> target);

    protected GraphNode(Tensor<float> target) {
        this.target = target;
    }

    public Tensor<float> execute() {
        apply(target);
        return target;
    }

    public static implicit operator GraphNode(Tensor<float> target) => new ValueNode(target);
}

public sealed class ValueNode: GraphNode {
    public ValueNode(Tensor<float> target): base(target) {}
    protected override void apply(Tensor<float> target) {}

    public override string ToString() => $"{target}";
}

public abstract class NonaryNode: GraphNode {
    protected override string node_name => "{nonary}";

    protected NonaryNode(Tensor<float> target): base(target) {}

    public override string ToString() => $"{target} = {node_name}({target.name})";
}

public abstract class UnaryNode: GraphNode {
    protected override string node_name => "{unary}";

    private readonly GraphNode other;

    protected UnaryNode(Tensor<float> target, GraphNode other): base(target) {
        this.other = other;
    }

    protected sealed override void apply(Tensor<float> target) {
        apply(target, other.execute());
    }

    protected abstract void apply(Tensor<float> target, Tensor<float> other);

    public override string ToString()
        => $"{target} = {node_name}({other})";
}

public abstract class BinaryNode: GraphNode {
    protected override string node_name => "{binary}";

    private readonly GraphNode left;
    private readonly GraphNode right;

    protected BinaryNode(Tensor<float> target, GraphNode left, GraphNode right): base(target) {
        this.left  = left;
        this.right = right;
    }

    protected sealed override void apply(Tensor<float> target) {
        apply(target, left.execute(), right.execute());
    }

    protected abstract void apply(Tensor<float> target, Tensor<float> left, Tensor<float> right);

    public override string ToString()
        => $"{target} = {left} {node_name} {right}";
}