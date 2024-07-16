namespace ML;

public interface IInitializer {
    void initialize_weights(Span<float> weights, int input_count, int output_count);
    void initialize_biases(Span<float> biases);
}

public sealed class RandomUniform: IInitializer {
    private readonly float min;
    private readonly float max;
    private readonly Random rand;

    public RandomUniform(Random rand, float min = -0.5f, float max = 0.5f) {
        this.rand = rand;
        this.min = min;
        this.max = max;
    }

    // Uniform distribution between min and max
    public void initialize_weights(Span<float> weights, int input_count, int output_count) {
        for (var i = 0; i < weights.Length; i++) {
            weights[i] = (float)(rand.NextDouble() * (max - min) + min);
        }
    }

    // Bias initialization to zero or small random value
    public void initialize_biases(Span<float> biases) {
        for (var i = 0; i < biases.Length; i++) {
            biases[i] = (float)(rand.NextDouble() * (max - min) + min);
        }
    }
}

public sealed class Xavier: IInitializer {
    private readonly Random rand;

    public Xavier(Random rand) => this.rand = rand;

    // Xavier initialization (Glorot) is useful for tanh and sigmoid activations
    public void initialize_weights(Span<float> weights, int input_count, int output_count) {
        var range = (float)Math.Sqrt(6.0f / (input_count + output_count));
        for (var i = 0; i < weights.Length; i++) {
            weights[i] = (float)(rand.NextDouble() * 2.0f * range - range);
        }
    }

    // Bias initialization to zero
    public void initialize_biases(Span<float> biases) {
        for (var i = 0; i < biases.Length; i++) {
            biases[i] = 0.0f;
        }
    }
}

public sealed class He: IInitializer {
    private readonly Random rand;

    public He(Random rand) => this.rand = rand;

    // He initialization is useful for ReLU and variants (leaky ReLU, etc.)
    public void initialize_weights(Span<float> weights, int input_count, int output_count) {
        var stddev = (float)Math.Sqrt(2.0f / input_count);
        for (var i = 0; i < weights.Length; i++) {
            weights[i] = (float)(rand.next_gaussian() * stddev);
        }
    }

    // Bias initialization to small constant value
    public void initialize_biases(Span<float> biases) {
        for (var i = 0; i < biases.Length; i++) {
            biases[i] = 0.1f;
        }
    }
}

public sealed class LeCun: IInitializer {
    private readonly Random rand;

    public LeCun(Random rand) => this.rand = rand;

    // LeCun initialization is useful for activation functions like SELU
    public void initialize_weights(Span<float> weights, int input_count, int output_count) {
        var stddev = (float)Math.Sqrt(1.0f / input_count);
        for (var i = 0; i < weights.Length; i++) {
            weights[i] = (float)(rand.next_gaussian() * stddev);
        }
    }

    // Bias initialization to zero
    public void initialize_biases(Span<float> biases) {
        for (var i = 0; i < biases.Length; i++) {
            biases[i] = 0.0f;
        }
    }
}

file static class RandomExtensions {
    // Extension method to generate a Gaussian distributed random number
    public static double next_gaussian(this Random rand, double mean = 0.0f, double stddev = 1.0f) {
        var u1 = 1.0 - rand.NextDouble();
        var u2 = 1.0 - rand.NextDouble();
        var randStdNormal = Math.Sqrt(-2.0f * Math.Log(u1)) * Math.Sin(2.0f * Math.PI * u2);
        return mean + stddev * randStdNormal;
    }
}