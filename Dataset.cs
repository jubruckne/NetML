namespace NetML;

public class Dataset {
    public readonly float[][] inputs;
    public readonly string[] labels;
    public readonly float[][] outputs;

    private Dataset(string name, float[][] inputs, float[][] outputs, string[] labels) {
        this.name    = name;
        this.inputs  = inputs;
        this.outputs = outputs;
        this.labels  = labels;
    }

    public string name { get; }
    public int sample_count => inputs.Length;

    public Record this[int sample] {
        get {
            for (var i = 0; i < outputs.Length; ++i)
                if (outputs[sample][i] != 0f)
                    return new(inputs[sample], i);

            throw new KeyNotFoundException(sample.ToString());
        }
    }

    public IEnumerable<Record> samples {
        get {
            for (var s = 0; s < inputs.Length; ++s) {
                for (var i = 0; i < outputs.Length; ++i)
                    if (outputs[s][i] != 0f)
                        yield return new(inputs[s], i);
            }
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
        var sample_count = inputs.Length;
        var p1_count     = (int)(sample_count * p);
        var p2_count     = sample_count - p1_count;

        return (new(
                    name + "_1",
                    inputs.Take(p1_count).ToArray(),
                    outputs.Take(p1_count).ToArray(),
                    labels
                   ),
                new(
                    name + "_2",
                    inputs.Skip(p1_count).Take(p2_count).ToArray(),
                    outputs.Skip(p1_count).Take(p2_count).ToArray(),
                    labels
                   )
            );
    }

    private static string read_url(string url) {
        using (HttpClient client = new HttpClient()) {
            var response = client.GetAsync(url).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            return response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        }
    }

    public static Dataset load_from_url(string url) {
        const string cache_dir = "../../../.cache/";

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

    public static Dataset load_from_file(string file_path, float sample_percent = 1f, string[]? labels = null) {
        var lines = File.ReadAllLines(file_path);
        var sample_count = (int)((lines.Length - 1) * sample_percent);
        var inputs       = new float[sample_count][];
        var outputs      = new float[sample_count][];

        for (var i = 0; i < sample_count; i++) {
            var values = lines[i + 1].Split(',').Select(float.Parse).ToArray();
            inputs[i] = values.Skip(1).Select(static x => x / 255.0f).ToArray(); // Normalize pixel values
            outputs[i] = new float[10];
            outputs[i][(int)values[0]] = 1.0f; // One-hot encode the label
        }

        return new(file_path, inputs, outputs, labels ?? ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
    }

    public readonly struct Record(float[] data, int label) {
        public override string ToString() {
            var s = "";
            for (var y = 0; y < 28; ++y) {
                for (var x = 0; x < 28; ++x) s += $"{format(data[x + y * 28])}";

                s += "\n";
            }

            return $"label: {label}\n{s.Replace("............................\n", "").Replace('.', ' ')}\n";
        }
    }
}