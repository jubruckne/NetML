using System.Diagnostics;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Running;
using Microsoft.Diagnostics.Runtime;
using NetML;
using NetML.Benchmark;
using NetML.ML;
using NetML.ML2;

var gguf = GGUFFile.from_file("../../../../../models/gpt2_127-fp32.gguf");

//Console.WriteLine(ggufFile.Header);
// Displaying some metadata
foreach (var kv in gguf.metadata) {
    Console.WriteLine($"{kv.Key} = {kv.Value}");
}

Console.WriteLine();

foreach (var t in gguf.tensors) {
    Console.WriteLine(t);
}

return;

Console.WriteLine($"Vector64<float> Supported: {Vector64<float>.IsSupported}, Accelerated: {Vector64.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<sbyte> Supported: {Vector128<sbyte>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<short> Supported: {Vector128<short>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<int> Supported: {Vector128<int>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> Supported: {Vector128<float>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> Supported: {Vector128<float>.IsSupported}, Accelerated: {Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<float> Supported: {Vector256<float>.IsSupported}, Accelerated: {Vector256.IsHardwareAccelerated}");
Console.WriteLine();

Context ctx = new Context();

var a = ctx.allocate_tensor<float>("a", [2, 2]);
var b = ctx.allocate_tensor<float>("b", [2, 2]);
var c = ctx.allocate_tensor<float>("c", [2, 2]);
var d = ctx.allocate_tensor<float>("d", [2]);
var x = ctx.allocate_tensor<float>("x", [2, 2]);

d[0] = 0.25f;
d[1] = 0.75f;

a[0, 0] = 1;
a[1, 0] = 1;
a[1, 1] = 1;
a[0, 1] = 1;

b[0, 0] = 2;
b[1, 0] = 2;
b[1, 1] = 2;
b[0, 1] = 2;

c[0, 0] = 3;
c[1, 0] = 3;
c[1, 1] = 3;
c[0, 1] = 3;




var g = Graph.with_target(x)
             .add(x, a, b)
             .multiply(x, c)
             .sigmoid(x)
    ;

d.print();
Console.WriteLine(g.ToString());
g.execute().print();

Console.WriteLine();
Console.WriteLine($"Context size: {ctx.allocated_memory}");

return;


//var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
var ds = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
//var mlp = new Network(ds.name, [ds.input_length, 256, 128, ds.output_length]);

var mlp = new Network(
                      ds.name,
                      [
                          Layer.dense<Operator.Tanh<float>>("l1", ds.input_length, 192),
                          Layer.dense<Operator.Sigmoid>("l5", 192, ds.output_length),
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

Console.WriteLine();
Console.WriteLine("Training...");
trainer.train(ds, 0.002551f, 16, 35);

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