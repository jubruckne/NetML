namespace NetML.ML;

public struct Layer: IDisposable {
    public Matrix weights { get; }
    public Vector biases { get; }

    private Vector outputs { get; }

    public Layer(int input_size, int output_size) {
        weights = new Matrix(output_size, input_size);
        biases = new Vector(output_size);
        outputs = new Vector(output_size);

        initialize_weights();
    }

    private void initialize_weights() {
        var rand = new Random();
        for (var i = 0; i < weights.rows; i++) {
            for (var j = 0; j < weights.columns; j++)
            {
                weights[i, j] = (float)rand.NextDouble() * 0.1f;
            }
        }

        for (var i = 0; i < biases.length; i++) {
            biases[i] = 0;
        }
    }

    public Vector forward(Vector inputs) {
        Matrix.multiply(weights, inputs, outputs);

        for (var i = 0; i < outputs.length; i++) {
            outputs[i] += biases[i];
        }

        sigmoid(outputs);

        return outputs;
    }

    public Vector backward(Vector inputs, Vector gradients, float learning_rate) {
        var derivatives = new Vector(gradients.length);
        var input_errors =  new Vector(derivatives.length);

        sigmoid_derivative(outputs, derivatives);

        Matrix.multiply(weights, derivatives, input_errors);

        weights.add_outer_product_weighted(derivatives, inputs, learning_rate);
        biases.add_weighted(derivatives, learning_rate);

        return input_errors;
    }

    private static void sigmoid(Vector value) {
        for (var i = 0; i < value.length; i++) {
            value[i] = sigmoid(value[i]);
        }
    }

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }

    private static void sigmoid_derivative(Vector value) {
        for (var i = 0; i < value.length; i++) {
            var sig = sigmoid(value[i]);
            value[i] = sig * (1f - sig);
        }
    }

    private static void sigmoid_derivative(Vector value, Vector result) {
        for (var i = 0; i < value.length; i++) {
            var sig = sigmoid(value[i]);
            result[i] = sig * (1f - sig);
        }
    }


    public void Dispose() {
        weights.Dispose();
        biases.Dispose();
        outputs.Dispose();
    }
}