using System.Collections;
using System.Runtime.CompilerServices;

namespace NetML;

public enum DatasetType {
    /// <summary>
    /// 32x32x3 pixels
    /// </summary>
    Cifar10_Test,

    /// <summary>
    /// 32x32x3 pixels
    /// </summary>
    Cifar10_Train
}

public sealed class Dataset: IEnumerable<(float[] inputs, int label, string name)> {
    public string[] labels { get; }
    public string name { get; }
    public int length => inputs.Count;

    public int input_length => inputs[0].Length;
    public int output_length => outputs[0].Length;

    private readonly List<float[]> inputs;
    private readonly List<float[]> outputs;
    private int[] indices;

    private const string cache_dir = "../../../.cache/";

    private Dataset(string name, List<float[]> inputs, List<float[]> outputs, string[] labels) {
        this.name    = name;
        this.inputs = inputs;
        this.outputs = outputs;
        this.labels  = labels;
        this.indices = Enumerable.Range(0, inputs.Count).ToArray();
    }

    public Sample this[int sample] {
        get {
            var idx = indices[sample];

            for (var i = 0; i < outputs.Count; ++i) {
                if (outputs[idx][i] != 0f) {
                    return new(inputs[idx], outputs[idx], i);
                }
            }

            throw new KeyNotFoundException(sample.ToString());
        }
    }

    private static char format(float intensity) {
        intensity *= 100f;

        if (intensity is < 0 or > 100)
            throw new ArgumentOutOfRangeException(nameof(intensity), "Intensity must be between 0 and 100.");

        if (intensity >= 85) return '\u2588';
        if (intensity >= 65) return '\u2593';
        if (intensity >= 45) return '\u2592';
        if (intensity >= 25) return '\u2591';
        /*if (intensity >= 50) return '▐';
        if (intensity >= 40) return '▄';
        if (intensity >= 30) return '▀';
        if (intensity >= 20) return '█';
        if (intensity >= 10) return '▒';*/
        return '.'; // 0-9%
    }

    public (Dataset d1, Dataset d2) split(float p) {
        var sample_count = inputs.Count;
        var p1_count     = (int)(sample_count * p);
        var p2_count     = sample_count - p1_count;

        return (new(
                    name + "_1",
                    inputs.Take(p1_count).ToList(),
                    outputs.Take(p1_count).ToList(),
                    labels
                   ),
                new(
                    name + "_2",
                    inputs.Skip(p1_count).Take(p2_count).ToList(),
                    outputs.Skip(p1_count).Take(p2_count).ToList(),
                    labels
                   )
            );
    }

    public void shuffle(Random rand)
        => rand.Shuffle(indices);

    public void reverse()
        => indices = indices.Reverse().ToArray();

    public static Dataset operator+(Dataset left, Dataset right) {
        return new Dataset(
                                     left.name + "+" + right.name,
                                     left.inputs.Concat(right.inputs).ToList(),
                                     left.outputs.Concat(right.outputs).ToList(),
                                     left.labels
                                    );
    }

    private static string read_url(string url) {
        using HttpClient client = new HttpClient();

        var response = client.GetAsync(url).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();
        return response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
    }

    public static Dataset load_from_url(string url) {
        if (!Directory.Exists(cache_dir)) {
            Directory.CreateDirectory(cache_dir);
        }

        var filename = $"{cache_dir}/{new Uri(url).Segments[^1]}";

        if (!File.Exists(filename)) {
            var data = read_url(url);
            File.WriteAllText(filename, data);
        }

        return load_from_file(filename);
    }

    public static Dataset load(DatasetType type) {
        if (type == DatasetType.Cifar10_Test) {
            return load_from_file(
                                  "cifar10_test.csv",
                                  1f,
                                  [
                                      "Airplane",
                                      "Automobile",
                                      "Bird",
                                      "Cat",
                                      "Deer",
                                      "Dog",
                                      "Frog",
                                      "Horse",
                                      "Ship",
                                      "Truck"
                                  ]
                                 );
        }

        if (type == DatasetType.Cifar10_Train) {
            return load_from_file(
                                  "cifar10_train.csv",
                                  1f,
                                  [
                                      "Airplane",
                                      "Automobile",
                                      "Bird",
                                      "Cat",
                                      "Deer",
                                      "Dog",
                                      "Frog",
                                      "Horse",
                                      "Ship",
                                      "Truck"
                                  ]
                                 );
        }

        throw new ArgumentException("Invalid dataset type");
    }

    public static Dataset load_from_file(string filename, float sample_percent = 1f, string[]? labels = null) {
        if (!File.Exists(filename)) {
            filename = $"{cache_dir}/{filename}";
        }

        var lines = File.ReadAllLines(filename);
        var sample_count = (int)(lines.Length * sample_percent);
        var inputs       = new float[sample_count][];
        var outputs      = new float[sample_count][];

        for (var i = 0; i < sample_count; i++) {
            var values = lines[i].Split(',').Select(float.Parse).ToArray();
            inputs[i] = values.Skip(1).Select(static x => x / 255.0f).ToArray(); // Normalize pixel values
            outputs[i] = new float[10];
            outputs[i][(int)values[0]] = 1.0f; // One-hot encode the label
        }

        return new(filename, inputs.ToList(), outputs.ToList(), labels ?? ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
    }

    public readonly ref struct Sample {
        public ReadOnlySpan<float> input { get;}
        public ReadOnlySpan<float> output { get; }
        public int label { get; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Sample(Span<float> input, Span<float> output, int label) {
            this.input = input;
            this.output = output;
            this.label = label;
        }

        public override string ToString() {
            var s = "";
            for (var y = 0; y < 28; ++y) {
                for (var x = 0; x < 28; ++x) s += $"{format(input[x + y * 28])}";

                s += "\n";
            }

            return $"label: {label}\n{s.Replace("............................\n", "").Replace('.', ' ')}\n";
        }
    }

    public IEnumerator<(float[] inputs, int label, string name)> GetEnumerator() {
        for (var i = 0; i < length; ++i) {
            var sample = this[i];
            yield return (sample.input.ToArray(), sample.label, labels[sample.label]);
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
        => GetEnumerator();
}