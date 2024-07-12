using System.Runtime.Intrinsics;
using NetML;
using NetML.ML;

Console.WriteLine($"AdvSimd.IsSupported: {AdvSimd.IsSupported}");
Console.WriteLine($"X86.Avx.IsSupported: {System.Runtime.Intrinsics.X86.Avx.IsSupported}");
Console.WriteLine($"X86.Avx2.IsSupported: {System.Runtime.Intrinsics.X86.Avx2.IsSupported}");
Console.WriteLine($"X86.Fma.IsSupported: {System.Runtime.Intrinsics.X86.Fma.IsSupported}");
Console.WriteLine($"X86.Sse41.IsSupported: {System.Runtime.Intrinsics.X86.Sse41.IsSupported}");
Console.WriteLine($"X86.Sse42.IsSupported: {System.Runtime.Intrinsics.X86.Sse42.IsSupported}");
Console.WriteLine();
Console.WriteLine($"Vector64<float> IsSupported: {Vector64<float>.IsSupported}, IsHardwareAccelerated:{Vector64.IsHardwareAccelerated}");
Console.WriteLine($"Vector128<float> IsSupported: {Vector128<float>.IsSupported}, IsHardwareAccelerated:{Vector128.IsHardwareAccelerated}");
Console.WriteLine($"Vector256<float> IsSupported: {Vector256<float>.IsSupported}, IsHardwareAccelerated:{Vector256.IsHardwareAccelerated}");
Console.WriteLine($"Vector512<float> IsSupported: {Vector512<float>.IsSupported}, IsHardwareAccelerated:{Vector512.IsHardwareAccelerated}");
Console.WriteLine();


//var ds = Dataset.load(DatasetType.Cifar10_Train);
//ds.shuffle(Random.Shared);
var ds =
   Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
//   + Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");

var mlp = new Network([ds.input_length, 12, ds.output_length], 8);
var trainer = new Trainer(mlp);

trainer.train(ds, 0.00317f, 16, 250);