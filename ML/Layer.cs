using System.Runtime.CompilerServices;

namespace NetML.ML;

public interface ILayer {
    string name { get; }
    int input_size { get; }
    int output_size { get; }

    bool is_input_layer => false;
    bool is_output_layer => false;
    bool is_trainable => false;

    Vector forward(Vector input);
}

public interface ITrainableLayer: ILayer {
    Vector backward(Vector output_gradients);
    void apply_gradients(float learning_rate, int batch_size);
    bool ILayer.is_trainable => true;
    void initialize_weights<TInitializer>(Random random) where TInitializer: IInitializer<float>;
}

public sealed class InputLayer: ILayer {
    public string name => "input";
    public bool is_input_layer => true;
    public int input_size { get; }
    public int output_size => input_size;

    public Vector forward(Vector input) => input;
    public Vector backward(Vector output_gradients) => output_gradients;

    public InputLayer(int input_size) {
        this.input_size = input_size;
    }
}

public sealed class OutputLayer: ILayer, IDisposable {
    public string name => "output";
    public bool is_output_layer => true;
    public int input_size => output_size;
    public int output_size { get; }

    private Vector gradient { get; }

    public Vector forward(Vector input) => input;

    public Vector backward(Vector predicted, Vector expected) {
        Vector.subtract_elementwise(expected, predicted, gradient);
        return gradient;
    }

    public OutputLayer(int output_size) {
        this.output_size = output_size;
        this.gradient = new("output.expected", output_size);
    }

    public void Dispose() => gradient.Dispose();
}

public abstract class Layer: ITrainableLayer, IDisposable {
    public string name { get; }
    public Matrix weights { get; }
    public Vector biases { get; }

    private readonly Vector input;
    private readonly Vector output;

    private readonly Vector output_errors;
    private readonly Vector output_derivatives;
    private readonly Vector input_gradients;

    private readonly Matrix weight_gradients;
    private readonly Vector bias_gradients;

    public int input_size => weights.input_count;
    public int output_size => weights.output_count;

    protected Layer(string name, int input_size, int output_size) {
        // output = rows
        // input = columns
        this.name = name;
        weights   = new($"{name}_weighs", output_size, input_size);
        biases    = new($"{name}_biases", output_size);

        Console.WriteLine($"Layer {name}... input_size: {input_size}, output_size: {output_size}");

        input  = new($"{name}_inputs", input_size);
        output = new($"{name}_outputs", output_size);
        //this.inputs  = new Matrix("inputs", batch_size, input_size);
        //this.outputs = new Matrix("outputs", batch_size, output_size);

        output_derivatives = new($"{name}_derivatives", output_size);
        output_errors      = new($"{name}_errors", output_size);
        input_gradients    = new($"{name}_input_gradients", input_size);

        weight_gradients = new($"{name}_weigh_gradients", output_size, input_size);
        bias_gradients   = new($"{name}_bias_gradients", output_size);
    }

    public void initialize_weights<TInitialization>(Random random) where TInitialization: IInitializer<float> {
        Operator.apply_inplace<TInitialization>(weights.as_span());
        Operator.apply_inplace<TInitialization>(biases.as_span());
        weight_gradients.clear();
        bias_gradients.clear();
    }

    /*public void initialize_weights(Random rand) {
        var scale = (float)Math.Sqrt(2.0 / weights.input_count);
        for (var output_idx = 0; output_idx < weights.output_count; output_idx++) {
            for (var input_idx = 0; input_idx < weights.input_count; input_idx++)
                weights[output_idx, input_idx] = (float)(rand.NextDouble() * 2 - 1) * scale;
        }

        for (var i = 0; i < biases.length; i++) biases[i] = 0; // Initialize biases to zero
    }*/

    public Vector forward(Vector input) {
        Metrics.Vector_Insert.Start();
        this.input.insert(input.as_span());
        Metrics.Vector_Insert.Stop();

        Metrics.Matrix_Multiply.Start();
        Matrix.multiply(weights, input, output);
        Metrics.Matrix_Multiply.Stop();

        Metrics.Vector_Add_Elementwise.Start();
        output.add_elementwise(biases);
        Metrics.Vector_Add_Elementwise.Stop();

        Metrics.Activation_Sigmoid.Start();

        apply_activation(output);

        Metrics.Activation_Sigmoid.Stop();

        return output;
    }

    protected abstract void apply_activation(Vector vector);

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

    public static ITrainableLayer dense<TActivation>(string name, int input_size, int output_size)
        where TActivation: IActivation<float>
        => new Layer<TActivation>(name, input_size, output_size);
}


[SkipLocalsInit]
public sealed class Layer<TActivation>: Layer
    where TActivation: IActivation<float> {

    public Layer(string name, int input_size, int output_size):
        base(name, input_size, output_size) {}

    protected override void apply_activation(Vector vector)
        => Operator.apply<TActivation>(vector.as_span(), vector.as_span());
}