using System.Runtime.InteropServices;

namespace NetML.ML;

public unsafe class Matrix: IDisposable {
    public int rows { get; }
    public int columns { get; }

    private readonly float* data;
    private readonly int length;

    public Matrix(int rows, int columns) {
        this.rows    = rows;
        this.columns = columns;
        this.length = rows * columns;
        this.data = (float*)NativeMemory.AlignedAlloc((UIntPtr)(length * sizeof(float)), 16);
    }

    public float this[int row, int column] {
        get => data[row * columns + column];
        set => data[row * columns + column] = value;
    }

    public void add_weighted(Matrix other, float weight) {
        for (var i = 0; i < length; i++) {
            data[i] += other.data[i] * weight;
        }
    }

    public static void multiply(Matrix left, Vector right, Vector result) {
        for (var i = 0; i < left.rows; i++) {
            float sum = 0;
            for (var j = 0; j < left.columns; j++) {
                sum += left[i, j] * right[j];
            }
            result[i] = sum;
        }
    }

    public void add_outer_product_weighted(Vector vec1, Vector vec2, float weight) {
        for (var i = 0; i < vec1.length; i++) {
            for (var j = 0; j < vec2.length; j++) {
                this[i, j] += vec1[i] * vec2[j] * weight;
            }
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

    ~Matrix() {
        ReleaseUnmanagedResources();
    }
}