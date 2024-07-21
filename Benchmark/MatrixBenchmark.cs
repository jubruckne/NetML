using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NetML.ML;

namespace NetML.Benchmark;

// var summary = BenchmarkRunner.Run<VectorLoadBenchmarks>();

public class SingleCoreConfig {
    public static IConfig get() {
        var cfg = DefaultConfig.Instance;

        cfg.WithOption(ConfigOptions.LogBuildOutput, true)
           .WithOption(ConfigOptions.KeepBenchmarkFiles, false)
           .WithOptions(ConfigOptions.StopOnFirstError)
            ;
        return cfg;
    }
}

[MemoryDiagnoser]
[CpuDiagnoser]
public unsafe class MatrixBenchmark {
    private Matrix left;
    private Vector right;
    private Vector result;

    [Params(16, 32, 64, 128, 256, 512)]
    public int N;

    [GlobalSetup]
    public void Setup() {
        left   = new Matrix("left", 16 * N, 128 * N);
        right  = new Vector("right", 128 * N);
        result = new Vector("result", 16 * N);

        for (var i = 0; i < left.linear_length; i++) {
            left.data[i]  = i;
        }

        for (var i = 0; i < right.length; i++) {
            right.data[i]  = i;
        }

        result.clear();
    }

    /*[Benchmark]
    public void MatVec_mul_csharp_scalar() {
        Matrix.multiply_scalar(left, right, result);
    }

    [Benchmark]
    public void MatVec_mul_csharp_vec() {
        Vector.use_accelerate = false;
        Matrix.multiply(left, right, result);
    }*/

    [Benchmark]
    public void MatVec_mul_csharp_vec_opt() {
        Vector.use_accelerate = false;
        Matrix.multiply_vec_opt(left, right, result);
    }


    [Benchmark]
    public void MatVec_mul_blas() {
         Vector.use_accelerate = true;
         Matrix.multiply(left, right, result);
     }
}