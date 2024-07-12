using System.Runtime.Intrinsics;
using NetML;
using NetML.ML;

Console.WriteLine($"Vector64<float> IsSupported: {Vector64<float>.IsSupported}, IsHardwareAccelerated:{Vector64.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> IsSupported: {Vector128<float>.IsSupported}, IsHardwareAccelerated:{Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<float> IsSupported: {Vector256<float>.IsSupported}, IsHardwareAccelerated:{Vector256.IsHardwareAccelerated}");
Console.WriteLine();

//var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
var ds =
   Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv")
   + Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv")
   + Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv")
   + Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");

var mlp = new Network([ds.input_length, 128, ds.output_length], 16);
var trainer = new Trainer(mlp);

trainer.train(ds, 0.00751f, 16, 9);