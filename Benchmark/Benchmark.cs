using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Attributes;

// var summary = BenchmarkRunner.Run<VectorLoadBenchmarks>();


public unsafe class VectorLoadBenchmarks {
    private float* data1;
    private float* data2;
    private float* data3;

    private const int Length = 1024 * 16 * 16 * 16 * 16;

    [GlobalSetup]
    public void Setup() {
        data1 = (float*)NativeMemory.AlignedAlloc((Length + 16) * sizeof(float), 16) + 2;
        data2 = (float*)NativeMemory.AlignedAlloc((Length + 16) * sizeof(float), 16) + 2;
        data3 = (float*)NativeMemory.AlignedAlloc((Length + 16) * sizeof(float), 16) + 3;

        for (var i = 0; i < Length; i++) {
            data1[i] = i;
            data2[i] = i;
            data3[i] = i;
        }
    }
/*
    [Benchmark]
    public void LoadVector128_load_aligned() {
        for (var i = 0; i < Length; i += Vector128<float>.Count) {
            var vector = Vector128.Load(data_aligned + i);
            vector *= 1.1f;
            vector.Store(data_aligned + i);
        }
    }

    [Benchmark]
    public void LoadVector128_load_unaligned() {
        for (var i = 0; i < Length; i += Vector128<float>.Count) {
            var vector = Vector128.Load(data_unaligned + i);
            vector *= 1.1f;
            vector.Store(data_unaligned + i);
        }
    }

    [Benchmark]
    public void LoadVector128_loadaligned_aligned() {
        for (var i = 0; i < Length; i += Vector128<float>.Count) {
            var vector = Vector128.LoadAligned(data_aligned + i);
            vector *= 1.1f;
            vector.StoreAligned(data_aligned + i);
        }
    }

    [Benchmark]
    public void LoadVector128_loadaligned_unaligned() {
        for (var i = 0; i < Length; i += Vector128<float>.Count) {
            var vector = Vector128.LoadAligned(data_unaligned + i);
            vector *= 1.1f;
            vector.StoreAligned(data_unaligned + i);
        }
    }
*/
    [Benchmark]
    public void LoadVector128_advsimd_load_aligned() {
        for (var i = 0; i < Length; i += 4) {
            var vector = Vector128.Load(data1 + i);
            vector.Store(data3 + i);
        }
    }

    [Benchmark]
    public void LoadVector128_advsimd_load_aligned_static() {
        var one = Vector128.Create(1.1f);

        for (var i = 0; i < Length; i += 4) {
            var vector = VectorExt.LoadVector(data2 + i);
            VectorExt.StoreVector(vector, data3 + i);
        }
    }


/*
    [Benchmark]
    public void LoadVector128_advsimd_load_unaligned() {
        for (var i = 0; i < Length; i += Vector128<float>.Count) {
            var vector = AdvSimd.LoadVector128(data_unaligned + i);
            vector *= 1.1f;
            AdvSimd.Store(data_unaligned + i, vector);
        }
    }
    */
}

public static unsafe class VectorExt {
    [MethodImpl(MethodImplOptions.AggressiveInlining), SkipLocalsInit]
    public static Vector128<float> LoadVector(float* ptr) {
        return *(Vector128<float>*)ptr;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining), SkipLocalsInit]
    public static void StoreVector(Vector128<float> vector, float* ptr) {
        *(Vector128<float>*)ptr = vector;
    }

}