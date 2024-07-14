using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NetML.ML;

[SkipLocalsInit]
public sealed class Trainer {
    public Network network { get; }

    public Trainer(Network network) {
        this.network = network;
    }

    public void train(Dataset dataset, float learning_rate, int mini_batch_size, int epochs) {
        Stopwatch sw = new();
        sw.Start();

        for (var epoch = 0; epoch < epochs; epoch++) {
            //if (epoch % 2 == 0) dataset.shuffle(network.random);

            float total_error = 0;

            for (var i = 0; i < dataset.length; ++i) {
                if (i % 10000 == 0)
                    Console.Write($"\rEpoch {epoch + 1}, sample {i:N0} / {dataset.length:N0}, lr={learning_rate:N4}");

                var input = dataset[i].input;
                var output = dataset[i].output;

                var predicted = network.forward(input);

                network.backward(predicted, output);

                if (i % mini_batch_size == 0) {
                    network.apply_gradients(learning_rate, mini_batch_size);
                }

                total_error += Vector.mean_squared_error(predicted, output);
            }

            if (dataset.length % mini_batch_size != 0) {
                network.apply_gradients(learning_rate, mini_batch_size);
            }

            Console.WriteLine($"\rEpoch {epoch + 1}, loss: {total_error:N0}, duration {sw.ElapsedMilliseconds:N0} ms       ");
        }

        sw.Stop();

        double total_samples = epochs * dataset.length;
        Console.WriteLine($"processed {total_samples} samples at {total_samples / sw.ElapsedMilliseconds:N2} samples/ms");
    }
}