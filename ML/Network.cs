using System.Runtime.CompilerServices;

namespace NetML.ML;

[SkipLocalsInit]
public sealed class Network: IDisposable {
    public Layer[] layers { get; }
    public Random random { get; }
    public int batch_size { get; }

    private readonly Vector gradient;

    public Network(Span<int> layer_sizes, int batch_size) {
        random = new Random(1337);

        this.batch_size = batch_size;
        layers = new Layer[layer_sizes.Length - 1];
        for (var i = 0; i < layers.Length; i++) {
            layers[i] = new Layer($"l{i}",layer_sizes[i], layer_sizes[i + 1], batch_size);
            layers[i].initialize_weights(random);
        }

        gradient = new Vector("expected", layers[^1].output_size);
    }

    public Vector forward(Vector input) {
        var x = input;
        foreach (var layer in layers) {
            x = layer.forward(x);
        }
        return x;
    }

    public void backward(Vector predicted, Vector expected) {
        var gradient = this.gradient;

        Vector.subtract_elementwise(expected, predicted, gradient);

        // Backward propagation
        for (var i = layers.Length - 1; i >= 0; i--) {
            gradient = layers[i].backward(gradient);
        }
    }

    public void apply_gradients(float learning_rate, int batch_size) {
        for (var i = layers.Length - 1; i >= 0; i--) {
            layers[i].apply_gradients(learning_rate, batch_size);
        }
    }

    public void Dispose() {
        gradient.Dispose();
    }
}