using System.Runtime.CompilerServices;

namespace NetML.ML;

[SkipLocalsInit]
public sealed class Layer: IDisposable {
    public string name { get; }
    public Matrix weights { get; }
    public Vector biases { get; }

    private readonly Vector inputs;
    private readonly Vector outputs;
    private readonly Vector output_errors;
    private readonly Vector output_derivatives;
    private readonly Vector input_gradients;

    public int input_size => weights.input_count;
    public int output_size => weights.output_count;

    public Layer(string name, int input_size, int output_size) {
        // output = rows
        // input = columns
        this.name = name;
        this.weights = new Matrix("weighs", output_size, input_size);
        this.biases = new Vector("biases", output_size);
        this.outputs = new Vector("outputs", output_size);

        this.inputs = new Vector("inputs", input_size);
        this.output_derivatives = new Vector("derivatives", output_size);
        this.output_errors = new Vector("errors", output_size);
        this.input_gradients = new Vector("input_gradients", input_size);
    }

    public void initialize_weights(Random rand) {
        var scale = (float)Math.Sqrt(2.0 / weights.input_count);
        for (var output_idx = 0; output_idx < weights.output_count; output_idx++) {
            for (var input_idx = 0; input_idx < weights.input_count; input_idx++)
            {
                weights[output_idx, input_idx] = (float)(rand.NextDouble() * 2 - 1) * scale;
            }
        }

        for (var i = 0; i < biases.length; i++) {
            biases[i] = 0; // Initialize biases to zero
        }
    }

    public Vector forward(Vector inputs) {
        this.inputs.load(inputs.as_span());

        // Console.WriteLine($"\n{name}_forward");

        Matrix.multiply(weights, inputs, outputs);

        outputs.add_elementwise(biases);

        ActivationFunctions.sigmoid(outputs);

        return outputs;
    }

    public Vector backward(Vector output_gradients, float learning_rate) {
        // Console.WriteLine($"\n{name}_backward");

        // Calculate the derivative of the outputs
        ActivationFunctions.sigmoid_derivative(outputs, output_derivatives);

        // Calculate the errors for each output node
        Vector.multiply_elementwise(output_gradients, output_derivatives, output_errors);
        Vector.dot(weights, output_gradients, input_gradients);

        weights.add_outer_product_weighted(output_errors, inputs, learning_rate);
        biases.add_elementwise_weighted(output_errors, learning_rate);

        return input_gradients;
    }

    public void Dispose() {
        weights.Dispose();
        biases.Dispose();
        outputs.Dispose();
        output_derivatives.Dispose();
        output_errors.Dispose();
        input_gradients.Dispose();
        inputs.Dispose();
    }
}