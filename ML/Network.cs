namespace NetML.ML;

public class Network {
    private readonly Layer[] layers;

    public Network(Span<int> layer_sizes) {
        layers = new Layer[layer_sizes.Length - 1];
        for (var i = 0; i < layers.Length; i++) {
            layers[i] = new Layer(layer_sizes[i], layer_sizes[i + 1]);
        }
    }

    public Vector forward(Vector inputs) {
        foreach (var layer in layers) {
            inputs = layer.forward(inputs);
        }
        return inputs;
    }

    public void backward(Vector inputs, Vector expected, float learning_rate) {
        var outputs = forward(inputs);
        var error = new Vector(expected.length);

        Vector.subtract(expected, outputs, error);

        for (var i = layers.Length - 1; i >= 0; i--) {
            error = layers[i].backward(inputs, error, learning_rate);
        }
    }
}