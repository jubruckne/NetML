using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NetML.ML;

[JsonConverter(typeof(NetworkConverter))]
[SkipLocalsInit]
public sealed class Network: IDisposable {
    public string name { get; }
    public List<ITrainableLayer> layers { get; }
    public InputLayer input_layer { get; }
    public OutputLayer output_layer { get; }
    public Random random { get; }

    public Network(string name, Span<int> layer_sizes) {
        this.name = name;
        this.random = new(1337);
        this.layers = new(layer_sizes.Length - 1);

        this.input_layer = new InputLayer(layer_sizes[0]);

        for (var i = 0; i < layer_sizes.Length - 1; i++) {
            var layer = new Layer<Operator.Sigmoid>($"l{i}", layer_sizes[i], layer_sizes[i + 1]);
            layers.Add(layer);
            layer.initialize_weights<Operator.RandomUniform<float>>(random);
        }

        this.output_layer = new OutputLayer(layer_sizes[^1]);
    }

    public Network(string name, List<ITrainableLayer> layers) {
        this.name = name;
        this.random = new(1337);
        this.layers = layers;
        this.input_layer = new InputLayer(layers[0].input_size);
        this.output_layer = new OutputLayer(layers[^1].output_size);

        foreach (var layer in layers) {
            layer.initialize_weights<Operator.RandomUniform<float>>(random);
        }
    }

    public Vector forward(Vector input) {
        var x = input_layer.forward(input);
        foreach (var layer in layers) {
            x = layer.forward(x);
        }
        return output_layer.forward(x);
    }

    public void backward(Vector predicted, Vector expected) {
        var gradient = output_layer.backward(predicted, expected);

        // Backward propagation
        for (var i = layers.Count - 1; i >= 0; i--) gradient = layers[i].backward(gradient);
    }

    public void apply_gradients(float learning_rate, int batch_size) {
        for (var i = layers.Count - 1; i >= 0; i--) layers[i].apply_gradients(learning_rate, batch_size);
    }

    public void save_to_file(string filename) {
        using var file = File.CreateText(filename);
        file.Write(this.to_json());
    }

    public static Network load_from_file(string filename) {
        using var file = File.Open(filename, FileMode.Open);
        return JsonSerializer.Deserialize<Network>(file)!;
    }

    public void Dispose() {
    }
}