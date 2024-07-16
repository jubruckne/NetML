using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NetML.ML;

public static class Serialization {
    public static string to_json(this Vector vector)
        => JsonSerializer.Serialize(vector);

    public static string to_json(this Matrix matrix)
        => JsonSerializer.Serialize(matrix);

    public static string to_json(this Layer layer)
        => JsonSerializer.Serialize(layer);

    public static string to_json(this Network network)
        => JsonSerializer.Serialize(network);
}

public class NetworkConverter: JsonConverter<Network> {
    private class Data {
        public string name { get; set; }
        public Layer[] layers { get; set; }
    }

    public override Network Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) {
        var data  = JsonSerializer.Deserialize<Data>(ref reader, options);
        var network = new Network(data.name, data.layers, 1);
        return network;
    }

    public override void Write(Utf8JsonWriter writer, Network value, JsonSerializerOptions options) {
        var data = new Data {
                                name   = value.name,
                                layers = value.layers
                            };

        JsonSerializer.Serialize(writer, data, options);
    }
}

public class LayerConverter: JsonConverter<Layer> {
    private class Data {
        public string name { get; set; }
        public int input_size { get; set; }
        public int output_size { get; set; }
        public Matrix weights { get; set; }
        public Vector biases { get; set; }
    }

    public override Layer Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) {
        var data   = JsonSerializer.Deserialize<Data>(ref reader, options);
        var layer = new Layer(data.name, data.input_size, data.output_size, 1);
        layer.weights.insert(data.weights);
        layer.biases.insert(data.biases);
        return layer;
    }

    public override void Write(Utf8JsonWriter writer, Layer value, JsonSerializerOptions options) {
        var data = new Data {
                                name   = value.name,
                                input_size = value.input_size,
                                output_size = value.output_size,
                                weights = value.weights,
                                biases = value.biases
                            };

        JsonSerializer.Serialize(writer, data, options);
    }
}

public class VectorConverter: JsonConverter<Vector> {
    private class Data {
        public string name { get; set; }
        public int length { get; set; }
        public float[] array { get; set; }
    }

    public override Vector Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) {
        var data   = JsonSerializer.Deserialize<Data>(ref reader, options);
        var vector = new Vector(data.name, data.length);
        vector.insert(data.array);
        return vector;
    }

    public override void Write(Utf8JsonWriter writer, Vector value, JsonSerializerOptions options) {
        var data = new Data {
                                name   = value.name,
                                length = value.length,
                                array  = value.as_span().ToArray()
                            };

        JsonSerializer.Serialize(writer, data, options);
    }
}

public class MatrixConverter: JsonConverter<Matrix> {
    private class Data {
        public string name { get; set; }
        public int output_count { get; set; }
        public int input_count { get; set; }
        public float[] array { get; set; }
    }

    public override Matrix Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) {
        var data   = JsonSerializer.Deserialize<Data>(ref reader, options);
        var matrix = new Matrix(data.name, data.output_count, data.input_count);
        matrix.insert(data.array);
        return matrix;
    }

    public override void Write(Utf8JsonWriter writer, Matrix value, JsonSerializerOptions options) {
        var vectorData = new Data {
                                      name         = value.name,
                                      output_count = value.output_count,
                                      input_count  = value.input_count,
                                      array        = value.as_span().ToArray()
                                  };

        JsonSerializer.Serialize(writer, vectorData, options);
    }
}