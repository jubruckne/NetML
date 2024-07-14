using System.Collections;
using System.Runtime.CompilerServices;
using NetML.ML;

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

public sealed class Dataset: IEnumerable<Dataset.Sample> {
    public string[] labels { get; }
    public string name { get; }
    public int length => m_inputs.output_count;

    public int input_length => m_inputs.input_count;
    public int output_length => m_outputs.input_count;

    private readonly Matrix m_inputs;
    private readonly Matrix m_outputs;

    private int[] indices;

    private const string cache_dir = "../../../.cache/";

    private Dataset(string name, List<float[]> inputs, List<float[]> outputs, string[] labels) {
        this.name    = name;
        this.labels  = labels;

        m_inputs = new Matrix("inputs", inputs.Count, inputs[0].Length);
        m_outputs = new Matrix("outputs", outputs.Count, outputs[0].Length);

        for (var i = 0; i < inputs.Count; i++) {
            m_inputs.insert(i, inputs[i]);
        }

        for (var i = 0; i < outputs.Count; i++) {
            m_outputs.insert(i, outputs[i]);
        }

        indices = Enumerable.Range(0, inputs.Count).ToArray();
    }

    public Sample this[int sample] {
        get {
            var idx = indices[sample];

            var input = m_inputs.view(0, idx);
            var output = m_outputs.view(0, idx);

            return new(input, output);
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
        var sample_count = m_inputs.output_count;
        var p1_count     = (int)(sample_count * p);
        var p2_count     = sample_count - p1_count;

        throw new NotImplementedException();
        /*return (new(
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
            */
    }

    public void shuffle(Random rand)
        => rand.Shuffle(indices);

    public void reverse()
        => indices = indices.Reverse().ToArray();

    public static Dataset operator+(Dataset left, Dataset right) {
        throw new NotImplementedException();
       /* return new Dataset(
                                     left.name + "+" + right.name,
                                     left.inputs.Concat(right.inputs).ToList(),
                                     left.outputs.Concat(right.outputs).ToList(),
                                     left.labels
                                    );
                                    */
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

    private float[] shift(float[] input, int width, int height, int shift_x, int shift_y) {
        var shifted = new float[input.Length];
        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                var new_x = x + shift_x;
                var new_y = y + shift_y;
                if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    shifted[new_x + new_y * width] = input[x + y * width];
                }
            }
        }
        return shifted;
    }

    private float[] rotate(float[] input, int width, int height, float angle) {
        var rotated = new float[input.Length];
        var rad     = MathF.PI * angle / 180.0f;
        var center_x = width / 2.0f;
        var center_y = height / 2.0f;

        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                var rel_x = x - center_x;
                var rel_y = y - center_y;
                var new_x = (int)(rel_x * float.Cos(rad) - rel_y * float.Sin(rad) + center_x);
                var new_y = (int)(rel_x * float.Sin(rad) + rel_y * float.Cos(rad) + center_y);

                if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    rotated[new_x + new_y * width] = input[x + y * width];
                }
            }
        }

        return rotated;
    }

    private float[] scale(float[] input, int width, int height, float scale) {
        var scaled  = new float[input.Length];
        var center_x = width / 2.0f;
        var center_y = height / 2.0f;

        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                var rel_x = (x - center_x) / scale + center_x;
                var rel_y = (y - center_y) / scale + center_y;
                var new_x = (int)float.Round(rel_x);
                var new_y = (int)float.Round(rel_y);

                if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    scaled[x + y * width] = input[new_x + new_y * width];
                }
            }
        }

        return scaled;
    }

    private float[] adjust_brightness(float[] input, float factor) {
        var adjusted = new float[input.Length];
        for (var i = 0; i < input.Length; i++) {
            adjusted[i] = float.Clamp(input[i] * factor, 0f, 1f);
        }
        return adjusted;
    }

    private float[] add_gaussian_noise(float[] input, float mean, float stddev, Random rand) {
        var noisy = new float[input.Length];
        for (var i = 0; i < input.Length; i++) {
            noisy[i] = input[i] + (float)next_gaussian(rand, mean, stddev);
        }
        return noisy;
    }

    private float[] add_noise(float[] input, float noise_level, Random rand) {
        var noisy = new float[input.Length];
        for (var i = 0; i < input.Length; i++) {
            noisy[i] = float.Clamp(input[i] + (rand.NextSingle() * 2f - 1f) * noise_level, 0f, 1f);
        }
        return noisy;
    }

    public readonly struct Sample {
        public Vector input { get;}
        public Vector output { get; }
        public int label { get; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Sample(Vector input, Vector output) {
            this.input = input;
            this.output = output;
            this.label = output.index_of_max_value();
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

    public IEnumerator<Sample> GetEnumerator() {
        for (var i = 0; i < length; ++i) {
            yield return this[i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
        => GetEnumerator();

    public static double next_gaussian(Random rand, float mean = 0f, float stdev = 1f) {
        // Using Box-Muller transform
        var u1            = 1.0f - rand.NextSingle(); // uniform(0,1] random doubles
        var u2            = 1.0f - rand.NextSingle();
        var rand_std_normal = float.Sqrt(-2.0f * float.Log(u1)) * float.Sin(2.0f * MathF.PI * u2); // random normal(0,1)
        return mean + stdev * rand_std_normal; // random normal(mean,stdDev^2)
    }
}