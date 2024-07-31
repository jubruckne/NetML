namespace NetML.ML2;

public static partial class TensorExtensions {
    public static void linear_projection(this Tensor<float> target,
                                         Tensor<float> x,
                                         Tensor<float> W,
                                         Tensor<float> b
    ) {
        Console.WriteLine("linear_projection");
        Console.WriteLine($"{target.friendly_name}\n= linear_projection(\n {x.friendly_name},\n {W.friendly_name},\n {b.friendly_name})");

        target.multiply(x, W);
        target.add(b);
    }
}