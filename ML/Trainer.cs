using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NetML.ML;

[SkipLocalsInit]
public sealed class Trainer {
    public Network network { get; }

    public Trainer(Network network) {
        this.network = network;
    }

    public void train(Dataset dataset, float learning_rate, int epochs) {
        Stopwatch sw = new();
        sw.Start();

        using var input = new Vector("input", network.layers[0].input_size);
        using var output = new Vector("output", network.layers[^1].output_size);

        for (var epoch = 0; epoch < epochs; epoch++) {
            float total_error = 0;

            for (var i = 0; i < dataset.length; ++i) {
                if (i % 1000 == 0)
                    Console.Write($"\rEpoch {epoch + 1}, sample {i:N0} / {dataset.length:N0}, lr={learning_rate:N4}");

                input.load(dataset[i].input);
                output.load(dataset[i].output);

                var predicted = network.forward(input);

                network.backward(predicted, output, learning_rate);
                total_error += output.Zip(predicted, static (target, output) => 0.5f * MathF.Pow(target - output, 2f)).Sum();
            }

            Console.WriteLine($"\rEpoch {epoch + 1}, loss: {total_error:N0}, duration {sw.ElapsedMilliseconds:N0} ms       ");
        }

        sw.Stop();

        double total_samples = epochs * dataset.length;
        Console.WriteLine($"processed {total_samples} samples at {total_samples / sw.ElapsedMilliseconds:N2} samples/ms");
    }


}