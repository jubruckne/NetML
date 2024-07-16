using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;

namespace NetML.ML;

[JsonConverter(typeof(LayerConverter))]
[SkipLocalsInit]
public sealed class Layer: IDisposable {
    public string name { get; }
    public Matrix weights { get; }
    public Vector biases { get; }

    private readonly Vector input;
    private readonly Vector output;
    //private readonly Matrix inputs;
    //private readonly Matrix outputs;

    private readonly Vector output_errors;
    private readonly Vector output_derivatives;
    private readonly Vector input_gradients;

    // Accumulators for gradients
    private readonly Matrix weight_gradients;
    private readonly Vector bias_gradients;

    public int input_size => weights.input_count;
    public int output_size => weights.output_count;

    public Layer(string name, int input_size, int output_size, int batch_size) {
        // output = rows
        // input = columns
        this.name = name;
        weights   = new("weighs", output_size, input_size);
        biases    = new("biases", output_size);

        Console.WriteLine($"Layer {name}... input_size: {input_size}, output_size: {output_size}");

        input  = new("inputs", input_size);
        output = new("outputs", output_size);
        //this.inputs  = new Matrix("inputs", batch_size, input_size);
        //this.outputs = new Matrix("outputs", batch_size, output_size);

        output_derivatives = new("derivatives", output_size);
        output_errors      = new("errors", output_size);
        input_gradients    = new("input_gradients", input_size);

        weight_gradients = new("weigh_gradients", output_size, input_size);
        bias_gradients   = new("bias_gradients", output_size);
        weight_gradients.clear();
        bias_gradients.clear();
    }

    public void initialize_weights(Random rand) {
        var scale = (float)Math.Sqrt(2.0 / weights.input_count);
        for (var output_idx = 0; output_idx < weights.output_count; output_idx++) {
            for (var input_idx = 0; input_idx < weights.input_count; input_idx++)
                weights[output_idx, input_idx] = (float)(rand.NextDouble() * 2 - 1) * scale;
        }

        for (var i = 0; i < biases.length; i++) biases[i] = 0; // Initialize biases to zero
    }

    public Vector forward(Vector input) {
        this.input.insert(input.as_span());

        Matrix.multiply(weights, input, output);

        output.add_elementwise(biases);

        ActivationFunctions.sigmoid(output);

        return output;
    }

    public Vector backward(Vector output_gradients) {
        // Console.WriteLine($"\n{name}_backward");

        // Calculate the derivative of the outputs
        ActivationFunctions.sigmoid_derivative(output, output_derivatives);

        // Calculate the errors for each output node
        Vector.multiply_elementwise(output_gradients, output_derivatives, output_errors);

        Vector.dot(weights, output_gradients, input_gradients);

        weight_gradients.add_outer_product(output_errors, input);
        bias_gradients.add_elementwise(output_errors);

        return input_gradients;
    }

    public void apply_gradients(float learning_rate, int batch_size) {
        weights.add_elementwise_weighted(weight_gradients, learning_rate);
        biases.add_elementwise_weighted(bias_gradients, learning_rate);

        weight_gradients.clear();
        bias_gradients.clear();
    }

    public void Dispose() {
        weights.Dispose();
        biases.Dispose();
        output.Dispose();
        output_derivatives.Dispose();
        output_errors.Dispose();
        input_gradients.Dispose();
        input.Dispose();
        weight_gradients.Dispose();
        bias_gradients.Dispose();
    }
}