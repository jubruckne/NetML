using System.Runtime.Intrinsics;
using NetML;
using NetML.ML;

Console.WriteLine($"Vector64<float> Supported: {Vector64<float>.IsSupported}, Accelerated: {Vector64.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> Supported: {Vector128<float>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<float> Supported: {Vector256<float>.IsSupported}, Accelerated: {Vector256.IsHardwareAccelerated}");
Console.WriteLine();

var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
//var ds  = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
var mlp = new Network("cifar10", [ds.input_length, 128, 64, 128, 64, ds.output_length], 16);
//var mlp = Network.load_from_file("../../../Models/cifar10.json");
Vector.use_accelerate = true;

var trainer = new Trainer(mlp);
// Correct: 9228 / 10000 (92,3 %)
// Correct: 9230 / 10000 (92,3 %)
// Correct: 9214 / 10000 (92,1 %)
// Correct: 9238 / 10000 (92,4 %)
//

Console.WriteLine();
Console.WriteLine("Training...");
trainer.train(ds, 0.00371f, 16, 35);
mlp.save_to_file("../../../Models/cifar10.json");

Console.WriteLine();
Console.WriteLine("Training finished.");
evaluate(mlp);

return;

void evaluate(Network network) {
    Console.WriteLine();
    Console.WriteLine("Evaluating.");

    //var dataset = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");
    var dataset = Dataset.load(DatasetType.Cifar10_Test);
    var correct = 0;

    foreach (var sample in dataset) {
        var prediction = network.forward(sample.input).index_of_max_value();
        if (prediction == sample.label) {
            correct++;
        } else {
            //Console.WriteLine($"Predicted: {prediction}, Actual: {sample}");
            //Console.WriteLine();
        }
    }

    Console.WriteLine();
    Console.WriteLine($"Correct: {correct} / {dataset.length} ({correct / (double)dataset.length:P1})");
}