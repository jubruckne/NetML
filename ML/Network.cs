using System.Runtime.CompilerServices;

namespace NetML.ML;

[SkipLocalsInit]
public sealed class Network: IDisposable {
    public Layer[] layers { get; }

    private readonly Vector gradient;

    public Network(Span<int> layer_sizes) {
        var rand = new Random(1337);

        layers = new Layer[layer_sizes.Length - 1];
        for (var i = 0; i < layers.Length; i++) {
            layers[i] = new Layer($"l{i}",layer_sizes[i], layer_sizes[i + 1]);
            layers[i].initialize_weights(rand);
        }

        gradient = new Vector("expected", layers[^1].output_size);
    }

    public Vector forward(Vector inputs) {
        var x = inputs;
        foreach (var layer in layers) {
            x = layer.forward(x);
        }
        return x;
    }

    public void backward(Vector predicted, Vector expected, float learning_rate) {
        var gradient = this.gradient;

        Vector.subtract_elementwise(expected, predicted, gradient);

        // Backward propagation
        for (var i = layers.Length - 1; i >= 0; i--) {
            gradient = layers[i].backward(gradient, learning_rate);
        }
    }

    public void Dispose() {
        gradient.Dispose();
    }
}