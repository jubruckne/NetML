using System.Runtime.Intrinsics;
using NetML;
using NetML.ML;

Console.WriteLine($"Vector64<float>.IsSupported: {Vector64<float>.IsSupported}");
Console.WriteLine($"Vector128<float>.IsSupported: {Vector128<float>.IsSupported}");

//var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
var ds =
   Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
//   + Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");

var mlp = new Network([ds.input_length, 12, ds.output_length], 8);
var trainer = new Trainer(mlp);

trainer.train(ds, 0.00317f, 16, 250);