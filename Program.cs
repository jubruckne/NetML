using System.Runtime.Intrinsics;
using System.Xml;
using NetML;
using NetML.ML;

Console.WriteLine($"Vector64<float> IsSupported: {Vector64<float>.IsSupported}, IsHardwareAccelerated:{Vector64.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> IsSupported: {Vector128<float>.IsSupported}, IsHardwareAccelerated:{Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<float> IsSupported: {Vector256<float>.IsSupported}, IsHardwareAccelerated:{Vector256.IsHardwareAccelerated}");
Console.WriteLine();

//var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
var ds = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
var mlp = new Network([ds.input_length, 128, ds.output_length], 16);

var trainer = new Trainer(mlp);
// Correct: 9228 / 10000 (92,3 %)
// Correct: 9230 / 10000 (92,3 %)
// Correct: 9214 / 10000 (92,1 %)
// Correct: 9238 / 10000 (92,4 %)
//

Console.WriteLine();
Console.WriteLine("Training...");
trainer.train(ds, 0.0047f, 8, 99);

Console.WriteLine();
Console.WriteLine("Training finished.");
Console.WriteLine();
Console.WriteLine("Evaluating.");

ds = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");
//ds = Dataset.load(DatasetType.Cifar10_Test);
var correct = 0;

foreach (var sample in ds) {
   var prediction = mlp.forward(sample.input).index_of_max_value();
   if (prediction == sample.label) {
      correct++;
   } else {
      Console.WriteLine($"Predicted: {prediction}, Actual: {sample}");
      Console.WriteLine();
   }
}

Console.WriteLine();
Console.WriteLine($"Correct: {correct} / {ds.length} ({correct / (double)ds.length:P1})");