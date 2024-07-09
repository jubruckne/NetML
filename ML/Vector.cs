using System.Runtime.InteropServices;

namespace NetML.ML;

public unsafe class Vector: IDisposable {
    public int length { get; }

    private readonly float* data;

    public Vector(int data_length) {
        this.length = data_length;
        this.data = (float*)NativeMemory.AlignedAlloc((UIntPtr)(data_length * sizeof(float)), 16);
    }

    public float this[int i] {
        get => data[i];
        set => data[i] = value;
    }

    public static void subtract(Vector left, Vector right, Vector result) {
        for (var i = 0; i < left.length; i++) {
            result[i] = left[i] - right[i];
        }
    }

    public void add_weighted(Vector other, float weight) {
        for (var i = 0; i < length; i++) {
            this[i] += other[i] * weight;
        }
    }

    public static void multiply_elementwise(Vector left, Vector right, Vector result) {
        for (var i = 0; i < left.length; i++) {
            result[i] = left[i] * right[i];
        }
    }

    private void ReleaseUnmanagedResources() {
        if (data != null) {
            NativeMemory.AlignedFree(data);
        }
    }

    public void Dispose() {
        ReleaseUnmanagedResources();
        GC.SuppressFinalize(this);
    }

    ~Vector() {
        ReleaseUnmanagedResources();
    }
}