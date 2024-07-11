using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace NetML.ML;

[SkipLocalsInit]
public sealed unsafe class Matrix: IDisposable {
    public string name { get; }
    public int output_count { get; } // rows
    public int input_count { get; }  // columns

    private readonly float* data;
    private readonly int linear_length;
    private int allocated;

    public Matrix(string name, int output_count, int input_count) {
        this.name   = $"{name}[rows={output_count}, col={input_count}]";
        this.output_count    = output_count;
        this.input_count = input_count;
        this.linear_length = output_count * input_count;
        this.data = (float*)NativeMemory.AlignedAlloc((UIntPtr)(linear_length * sizeof(float)), 16);
        this.allocated = 1;
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

    public static void multiply(Matrix left, Vector right, Vector result) {
        if (left.input_count != right.length)
            throw new Exception($"{right.name} must have length of {left.name} columns!");
        if (left.output_count != result.length)
            throw new Exception($"{result.name} must have length of {left.name} rows!");

        var left_ptr = left.data;
        var right_ptr = right.get_pointer();
        var result_ptr = result.get_pointer();

        for (var i = 0; i < left.output_count; i++) {
            var sum = Vector128<float>.Zero;

            // Process 4 elements at a time
            for (var j = 0; j < left.input_count; j += 4) {
                var left_vec = Vector128.LoadAligned(left_ptr + i * left.input_count + j);
                var right_vec = Vector128.LoadAligned(right_ptr + j);
                sum = AdvSimd.FusedMultiplyAdd(sum, left_vec, right_vec);
            }

            // Sum the 4 elements of the vector
            result_ptr[i] = Vector128.Sum(sum);
        }
    }

    public void add_outer_product_weighted(Vector left, Vector right, float weight) {
        var left_ptr   = left.get_pointer();
        var right_ptr  = right.get_pointer();

        for (var i = 0; i < left.length; i++) {
            var left_value = Vector128.Create(left_ptr[i] * weight);

            for (var j = 0; j < right.length; j += 4) {
                var right_vec = Vector128.LoadAligned(right_ptr + j);
                var v = Vector128.LoadAligned(data + i * right.length + j);
                var p = AdvSimd.FusedMultiplyAdd(v, right_vec, left_value);
                p.StoreAligned(data + i * right.length + j);
            }
        }
    }

    public static Matrix outer_product_weighted(Vector left, Vector right, float weight) {
        var left_length  = left.length;
        var right_length = right.length;

        var result = new Matrix($"{left.name} x {right.name}", left_length, right_length);

        var left_ptr   = left.get_pointer();
        var right_ptr  = right.get_pointer();
        var result_ptr = result.data;

        for (var i = 0; i < left_length; i++) {
            var left_value = left_ptr[i] * weight;

            for (var j = 0; j < right_length; j += 4) {
                var right_vec = Vector128.LoadAligned(right_ptr + j);
                var product  = right_vec * left_value;
                product.StoreAligned(result_ptr + i * right_length + j);
            }
        }

        return result;
    }

    public Matrix transpose() {
        var transposed = new Matrix(name + "[transposed]", input_count, output_count);

        for (var o = 0; o < output_count; o++) {
            for (var i = 0; i < input_count; i++) {
                transposed[i, o] = this[o, i];
            }
        }

        return transposed;
    }

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* get_pointer() => data;
}