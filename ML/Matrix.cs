using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NetML.ML;

[SkipLocalsInit]
public sealed unsafe class Matrix: IDisposable {
    public string name { get; }
    public int output_count { get; } // rows
    public int input_count { get; }  // columns
    public float* data { get; }
    public int linear_length { get; }

    private int allocated;

    public Matrix(string name, int output_count, int input_count) {
        this.name   = $"{name}[rows={output_count}, col={input_count}]";
        this.output_count    = output_count;
        this.input_count = input_count;
        this.linear_length = output_count * input_count;

        if(linear_length % 4 != 0) throw new ArgumentException($"length must be divisible by 4: {linear_length}");

        this.data = (float*)NativeMemory.AlignedAlloc((UIntPtr)(linear_length * sizeof(float)), 16);
        this.allocated = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void load(ReadOnlySpan<float> data) {
        if(data.Length != linear_length) throw new IndexOutOfRangeException($"{data.Length} != {linear_length}");
        data.CopyTo(as_span());
    }

    public float this[int output_idx, int input_idx] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get {
            if((uint)output_idx >= output_count) throw new IndexOutOfRangeException($"row {output_idx} >= {output_count}");
            if((uint)input_idx >= input_count) throw new IndexOutOfRangeException($"columns {input_idx} >= {input_count}");

            return data[output_idx * input_count + input_idx];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            if((uint)output_idx >= output_count) throw new IndexOutOfRangeException($"row {output_idx} >= {output_count}");
            if((uint)input_idx >= input_count) throw new IndexOutOfRangeException($"columns {input_idx} >= {input_count}");

            data[output_idx * input_count + input_idx] = value;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector view(int dim, int idx) {
        if(dim == 0)
            return new Vector($"{name}[{idx}]", output_count, data + idx * input_count);

        throw new NotImplementedException($"dim > 0 not implemented!");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void clear() => as_span().Clear();

    public void add_elementwise_weighted(Matrix other, float weight) {
        if (input_count != other.input_count) throw new Exception($"input_count != other.input_count");
        if (output_count != other.output_count) throw new Exception($"output_count != other.output_count");

        var linear_length = this.linear_length;
        var data_ptr = data;
        var other_ptr = other.data;

        for (var i = 0; i < linear_length; i += 4) {
            var v = Vector128.LoadAligned(data_ptr + i);
            var o = Vector128.LoadAligned(other_ptr + i);
            var m = v + o * weight;
            m.StoreAligned(data_ptr + i);
        }
    }

    public void add_elementwise(Matrix other) {
        if (input_count != other.input_count) throw new Exception($"input_count != other.input_count");
        if (output_count != other.output_count) throw new Exception($"output_count != other.output_count");

        for (var i = 0; i < linear_length; i += 4) {
            var v = Vector128.LoadAligned(data + i);
            var o = Vector128.LoadAligned(other.data + i);
            var m = v + o;
            m.StoreAligned(data + i);
        }
    }

    public void multiply_elementwise(float x) {
        if(linear_length % 4 != 0) throw new Exception($"linear_length % 4 != 0");

        for (var i = 0; i < linear_length; i += 4) {
            var v = Vector128.LoadAligned(data + i);
            var m = v * x;
            m.StoreAligned(data + i);
        }
    }

    public void divide_elementwise(float x) {
        if(linear_length % 4 != 0) throw new Exception($"linear_length % 4 != 0");

        for (var i = 0; i < linear_length; i += 4) {
            var v = Vector128.LoadAligned(data + i);
            var m = v / x;
            m.StoreAligned(data + i);
        }
    }

    public static void multiply(Matrix left, Vector right, Vector result) {
        if (left.input_count != right.length)
            throw new Exception($"{right.name} must have length of {left.name} columns!");
        if (left.output_count != result.length)
            throw new Exception($"{result.name} must have length of {left.name} rows!");

        var left_ptr = left.data;
        var right_ptr = right.data;
        var result_ptr = result.data;

        var output_length = left.output_count;
        var input_length = left.input_count;

        for (var i = 0; i < output_length; i++) {
            var sum = Vector128<float>.Zero;

            for (var j = 0; j < input_length; j += 4) {
                var left_vec = Vector128.LoadAligned(left_ptr + i * input_length + j);
                var right_vec = Vector128.LoadAligned(right_ptr + j);
                sum = Vector128.FusedMultiplyAdd(sum, left_vec, right_vec);
            }

            result_ptr[i] = Vector128.Sum(sum);
        }
    }

    public void add_outer_product(Vector left, Vector right) {
        var left_ptr  = left.data;
        var right_ptr = right.data;
        var data_ptr = data;

        var left_length = left.length;
        var right_length = right.length;

        for (var i = 0; i < left_length; i++) {
            var left_value = Vector128.Create(left_ptr[i]);

            for (var j = 0; j < right_length; j += 4) {
                var right_vec = Vector128.LoadAligned(right_ptr + j);
                var v         = Vector128.LoadAligned(data_ptr + i * right_length + j);
                var p         = Vector128.FusedMultiplyAdd(v, right_vec, left_value);
                p.StoreAligned(data_ptr + i * right_length + j);
            }
        }
    }

    public void add_outer_product_weighted(Vector left, Vector right, float weight) {
        var left_ptr  = left.data;
        var right_ptr = right.data;
        var data_ptr  = data;

        var left_length  = left.length;
        var right_length = right.length;

        for (var i = 0; i < left_length; i++) {
            var left_value = Vector128.Create(left_ptr[i] * weight);

            for (var j = 0; j < right_length; j += 4) {
                var right_vec = Vector128.LoadAligned(right_ptr + j);
                var v         = Vector128.LoadAligned(data_ptr + i * right_length + j);
                var p         = Vector128.FusedMultiplyAdd(v, right_vec, left_value);
                p.StoreAligned(data_ptr + i * right_length + j);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> as_span() => new Span<float>(data, linear_length);

    public override string ToString()
        => name;

    private void ReleaseUnmanagedResources() {
        if (Interlocked.CompareExchange(ref allocated, 0, 1) == 1) {
           //  Console.WriteLine($"releasing matrix {name}");
            NativeMemory.AlignedFree(data);
        }
    }

    public void Dispose() {
        if (allocated == 1) {
            ReleaseUnmanagedResources();
        }
        GC.SuppressFinalize(this);
    }

    ~Matrix() {
        if (allocated == 1) {
            ReleaseUnmanagedResources();
        }
    }
}