using System.Numerics.Tensors;
using Dia2Lib;

namespace NetML.ML2;

public sealed class Add_float_float_float: BinaryNode {
    protected override string node_name => "+";

    public Add_float_float_float(Tensor<float> target, GraphNode left, GraphNode right): base(target, left, right) {}

    protected override void apply(Tensor<float> target, Tensor<float> left, Tensor<float> right) {
        var dest = target.as_span();
        var l = left.as_readonly_span();
        var r = right.as_readonly_span();
        TensorPrimitives.Add(l, r, dest);
    }
}

public sealed class Multiply_float_float_float: BinaryNode {
    protected override string node_name => "*=";

    public Multiply_float_float_float(Tensor<float> target, GraphNode left, GraphNode right): base(target, left, right) {}

    protected override void apply(Tensor<float> target, Tensor<float> left, Tensor<float> right) {
        var dest = target.as_span();
        var l = left.as_readonly_span();
        var r = right.as_readonly_span();
        TensorPrimitives.Add(l, r, dest);
    }
}

public sealed class Sigmoid_float_float: UnaryNode {
    protected override string node_name => "sigmoid";

    public Sigmoid_float_float(Tensor<float> target, GraphNode other): base(target, other) {}

    protected override void apply(Tensor<float> target, Tensor<float> other) {
        TensorPrimitives.Sigmoid(other.as_readonly_span(), target.as_span());
    }
}

public sealed class Embedding: BinaryNode {
    protected override string node_name => "embedding";

    public Embedding(Tensor<float> target, GraphNode wte, GraphNode input): base(target, wte, input) {}

    protected override void apply(Tensor<float> target, Tensor<float> left, Tensor<float> right) {

    }
}