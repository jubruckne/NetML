using System.Diagnostics;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Running;
using NetML;
using NetML.Benchmark;
using NetML.ML;

Console.WriteLine($"Vector64<float> Supported: {Vector64<float>.IsSupported}, Accelerated: {Vector64.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<sbyte> Supported: {Vector128<sbyte>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<short> Supported: {Vector128<short>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<int> Supported: {Vector128<int>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> Supported: {Vector128<float>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> Supported: {Vector128<float>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<float> Supported: {Vector256<float>.IsSupported}, Accelerated: {Vector256.IsHardwareAccelerated}");
Console.WriteLine();


//var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
var ds = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
//var mlp = new Network(ds.name, [ds.input_length, 256, 128, ds.output_length]);

var mlp = new Network(
                      ds.name,
                      [
                          Layer.dense<Operator.Sigmoid>("l1", ds.input_length, 128),
                          Layer.dense<Operator.Sigmoid>("l2", 128, ds.output_length),
                      ]
                     );

//var mlp = Network.load_from_file($"../../../Models/{ds.name}.json");
Vector.use_accelerate = true;

Console.WriteLine(mlp.layers.Count);
Console.WriteLine(mlp.layers[0].input_size);
Console.WriteLine(mlp.layers[1].input_size);

//mlp.layers[0].activation = ActivationFunction.ReLU;
//mlp.layers[1].activation = ActivationFunction.Sigmoid;


var trainer = new Trainer(mlp);
// Correct: 9228 / 10000 (92,3 %)
// Correct: 9230 / 10000 (92,3 %)
// Correct: 9214 / 10000 (92,1 %)
// Correct: 9238 / 10000 (92,4 %)
//

Console.WriteLine();
Console.WriteLine("Training...");
trainer.train(ds, 0.00371f, 16, 5);

Metrics.print();

mlp.save_to_file($"../../../Models/{ds.name}.json");

Console.WriteLine();
Console.WriteLine("Training finished.");
evaluate(mlp);

return;

void evaluate(Network network) {
    Console.WriteLine();
    Console.WriteLine("Evaluating.");

    var dataset = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");
    //var dataset = Dataset.load(DatasetType.Cifar10_Test);
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