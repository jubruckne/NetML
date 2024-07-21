using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Text.Json.Serialization;

namespace NetML.ML;

[JsonConverter(typeof(MatrixConverter))]
[SkipLocalsInit]
public sealed unsafe class Matrix: IDisposable, ITensor<float> {
    public string name { get; }
    public int[] shape { get; }

    public int output_count { get; } // rows
    public int input_count { get; }  // columns
    public float* data { get; }
    public int linear_length { get; }

    private int allocated;

    public Matrix(string name, int output_count, int input_count) {
        this.name         = name;
        this.output_count = output_count;
        this.input_count  = input_count;
        this.linear_length = output_count * input_count;
        this.shape = [output_count, input_count];

        if ((nuint)input_count % 2 != 0)
            throw new ArgumentException($"{ToString()}: output_count must be a multiple of 2!");
        if ((nuint)input_count % 2 != 0)
            throw new ArgumentException($"{ToString()}: input_count must be a multiple of 2!");
        if (linear_length % 4 != 0) throw new ArgumentException($"length must be divisible by 4: {linear_length}");

        this.data      = (float*)NativeMemory.AlignedAlloc((UIntPtr)(linear_length * sizeof(float)), 16);
        this.allocated = 1;
    }

    public void insert(ReadOnlySpan<float> data) {
        if (data.Length != linear_length) throw new IndexOutOfRangeException($"{data.Length} != {linear_length}");
        data.CopyTo(as_span());
    }

    public void insert(int output_idx, ReadOnlySpan<float> data) {
        if (output_idx < 0 || output_idx >= output_count) throw new IndexOutOfRangeException(nameof(output_idx));
        if (data.Length != input_count) throw new IndexOutOfRangeException($"{data.Length} != {input_count}");
        data.CopyTo(as_span(output_idx, 1));
    }

    public void insert(int output_idx, Matrix other) {
        var other_span = other.as_span();
        var span       = as_span(output_idx, other.output_count);
        other_span.CopyTo(span);
    }

    public void insert(Matrix other) {
        other.as_readonly_span().CopyTo(as_span());
    }

    public float this[int output_idx, int input_idx] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get {
            if ((uint)output_idx >= output_count)
                throw new IndexOutOfRangeException($"row {output_idx} >= {output_count}");
            if ((uint)input_idx >= input_count)
                throw new IndexOutOfRangeException($"columns {input_idx} >= {input_count}");

            return data[output_idx * input_count + input_idx];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            if ((uint)output_idx >= output_count)
                throw new IndexOutOfRangeException($"row {output_idx} >= {output_count}");
            if ((uint)input_idx >= input_count)
                throw new IndexOutOfRangeException($"columns {input_idx} >= {input_count}");

            data[output_idx * input_count + input_idx] = value;
        }
    }

    float ITensor<float>.this[ReadOnlySpan<int> indices] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this[indices[0], indices[1]];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set =>  this[indices[0], indices[1]] = value;
    }

    float ITensor<float>.this[params int[] indices] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this[indices[0], indices[1]];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set =>  this[indices[0], indices[1]] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector view(int dim, int index) {
        if (dim != 0) throw new NotImplementedException("dim > 0 not implemented!");
        if (index < 0 || index >= output_count) throw new IndexOutOfRangeException($"row {index} >= {output_count}");

        return new("view", input_count, data + index * input_count);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void clear() {
        as_span().Clear();
    }

    public void add_elementwise_weighted(Matrix other, float weight) {
        if (input_count != other.input_count) throw new($"input_count != other.input_count");
        if (output_count != other.output_count) throw new($"output_count != other.output_count");

        if (Vector.use_accelerate) {
            Accelerate.vDSP_vsma(other.data, 1, &weight, data, 1, data, 1, (uint)linear_length);
        } else {
            var linear_length = this.linear_length;
            var data_ptr      = data;
            var other_ptr     = other.data;

            for (var i = 0; i < linear_length; i += 4) {
                var v = Vector128.Load(data_ptr + i);
                var o = Vector128.Load(other_ptr + i);
                var m = v + o * weight;
                m.Store(data_ptr + i);
            }
        }
    }

    public static void multiply(Matrix left, Matrix right, Matrix result) {
        if (left.input_count != right.output_count)
            throw new($"{right.name} must have number of rows equal to the number of columns of {left.name}!");
        if (left.output_count != result.output_count || right.input_count != result.input_count)
            throw new($"{result.name} must have the same dimensions as the product of {left.name} and {right.name}!");

        var left_ptr   = left.data;
        var right_ptr  = right.data;
        var result_ptr = result.data;

        var rows       = left.output_count;
        var cols       = right.input_count;
        var common_dim = left.input_count;

        for (var i = 0; i < rows; i++) {
            for (var j = 0; j < cols; j++) {
                var sum = Vector128<float>.Zero;

                for (var k = 0; k < common_dim; k += 4) {
                    var left_vec  = Vector128.Load(left_ptr + i * common_dim + k);
                    var right_vec = Vector128.Load(right_ptr + k * cols + j);
                    sum = Vector128.FusedMultiplyAdd(left_vec, right_vec, sum);
                }

                result_ptr[i * cols + j] = Vector128.Sum(sum);
            }
        }
    }

    public static void multiply_scalar(Matrix left, Vector right, Vector result) {
        var output_length = left.output_count;
        var input_length  = left.input_count;

        if (input_length != right.length)
            throw new($"{right.name} must have length of {left.name} columns!");
        if (output_length != result.length)
            throw new($"{result.name} must have length of {left.name} rows!");


        var left_ptr   = left.data;
        var right_ptr  = right.data;
        var result_ptr = result.data;

        for (var i = 0; i < output_length; i++) {
            var sum = 0f; //Vector128<float>.Zero;

            for (var j = 0; j < input_length; j++) {
                var left_vec  = left_ptr [i * input_length + j];
                var right_vec = right_ptr [j];
                sum += left_vec * right_vec; //Vector128.FusedMultiplyAdd(left_vec, right_vec, sum);
            }

            result_ptr[i] = sum; //Vector128.Sum(sum);
        }
    }

    public static void multiply_vec_opt(Matrix left, Vector right, Vector result) {
        var output_length = left.output_count;
        var input_length  = left.input_count;

        if (input_length != right.length)
            throw new($"{right.name} must have length of {left.name} columns!");
        if (output_length != result.length)
            throw new($"{result.name} must have length of {left.name} rows!");

        if (Vector.use_accelerate) {
            const int CblasRowMajor = 101;
            const int CblasNoTrans  = 111;
            float     alpha         = 1.0f;
            float     beta          = 0.0f;

            Accelerate.cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                   left.output_count, left.input_count,
                                   alpha, left.data, left.input_count,
                                   right.data, 1, beta, result.data, 1);
        } else {
            var left_ptr   = left.data;
            var right_ptr  = right.data;
            var result_ptr = result.data;

            for (var i = 0; i < output_length; i++) {
                var sum1 = Vector128<float>.Zero;
                var sum2 = Vector128<float>.Zero;
                var sum3 = Vector128<float>.Zero;
                var sum4 = Vector128<float>.Zero;

                for (var j = 0; j < input_length; j += 16) {
                    var (l1, l2)  = AdvSimd.Arm64.Load2xVector128(left_ptr + i * input_length + j);
                    var (l3, l4)  = AdvSimd.Arm64.Load2xVector128(left_ptr + i * input_length + (j + 8));
                    var (r1, r2) = AdvSimd.Arm64.Load2xVector128(right_ptr + j);
                    var (r3, r4) = AdvSimd.Arm64.Load2xVector128(right_ptr + (j + 8));

                    sum1 += l1 * r1;
                    sum2 += l2 * r2;
                    sum3 += l3 * r3;
                    sum4 += l4 * r4;
                }

                result_ptr[i] = Vector128.Sum(sum1 + sum2 + sum3 + sum4); //Vector128.Sum(sum);
            }
        }
    }

    public static void multiply(Matrix left, Vector right, Vector result) {
        var output_length = left.output_count;
        var input_length  = left.input_count;

        if (input_length != right.length)
            throw new($"{right.name} must have length of {left.name} columns!");
        if (output_length != result.length)
            throw new($"{result.name} must have length of {left.name} rows!");

        if (Vector.use_accelerate) {
            const int CblasRowMajor = 101;
            const int CblasNoTrans  = 111;
            float     alpha         = 1.0f;
            float     beta          = 0.0f;

            Accelerate.cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                   left.output_count, left.input_count,
                                   alpha, left.data, left.input_count,
                                   right.data, 1, beta, result.data, 1);
        } else {
            var left_ptr   = left.data;
            var right_ptr  = right.data;
            var result_ptr = result.data;

            for (var i = 0; i < output_length; i++) {
                var sum = 0f; //Vector128<float>.Zero;

                for (var j = 0; j < input_length; j += 4) {
                    var left_vec  = Vector128.Load(left_ptr + i * input_length + j);
                    var right_vec = Vector128.Load(right_ptr + j);
                    sum += Vector128.Dot(left_vec, right_vec); //Vector128.FusedMultiplyAdd(left_vec, right_vec, sum);
                }

                result_ptr[i] = sum; //Vector128.Sum(sum);
            }
        }
    }

    public void add_outer_product(Vector left, Vector right) {
        var left_ptr  = left.data;
        var right_ptr = right.data;
        var data_ptr  = data;

        var left_length  = left.length;
        var right_length = right.length;

        if (Vector.use_accelerate) {
            const int CblasRowMajor = 101;
            const int CblasNoTrans  = 111;
            float     alpha         = 1.0f;
            float     beta          = 1.0f;

            Accelerate.cblas_sgemm(
                                   CblasRowMajor,
                                   CblasNoTrans,
                                   CblasNoTrans,
                                   left.length,
                                   right.length,
                                   1,
                                   alpha,
                                   left.data,
                                   1,
                                   right.data,
                                   right.length,
                                   beta,
                                   data,
                                   input_count
                                  );
        } else {
            for (var i = 0; i < left_length; i++) {
                var left_value = Vector128.Create(left_ptr[i]);

                for (var j = 0; j < right_length; j += 4) {
                    var right_vec = Vector128.Load(right_ptr + j);
                    var v         = Vector128.Load(data_ptr + i * right_length + j);
                    var p         = Vector128.FusedMultiplyAdd(right_vec, left_value, v);
                    p.Store(data_ptr + i * right_length + j);
                }
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
                var right_vec = Vector128.Load(right_ptr + j);
                var v         = Vector128.Load(data_ptr + i * right_length + j);
                var p         = Vector128.FusedMultiplyAdd(right_vec, left_value, v);
                p.Store(data_ptr + i * right_length + j);
            }
        }
    }

    public static Matrix concatenate(string name, Matrix left, Matrix right) {
        if (left.output_count != right.output_count)
            throw new ArgumentException($"{left.name} output_count must equal {right.name} output_count!");

        var matrix = new Matrix(name, left.output_count + right.output_count, left.input_count);

        matrix.insert(0, left);
        matrix.insert(left.output_count + 1, right);

        return matrix;
    }

    public Span<float> as_span()
        => new(data, linear_length);

    public ReadOnlySpan<float> as_readonly_span()
        => new(data, linear_length);

    public Span<float> as_span(int output_idx, int rows)
        => new(data + output_idx * input_count, rows * input_count);

    public override string ToString() {
        return $"{name}[rows={output_count}, col={input_count}]";
    }

    private void ReleaseUnmanagedResources() {
        var alloc = Interlocked.CompareExchange(ref allocated, 0, 1);
        if (alloc == 1)
            // Console.WriteLine($"releasing matrix {name}");
            NativeMemory.AlignedFree(data);
    }

    public void Dispose() {
        if (allocated == 0) return;
        ReleaseUnmanagedResources();
        GC.SuppressFinalize(this);
    }

    ~Matrix() {
        if (allocated == 0) return;
        ReleaseUnmanagedResources();
    }
}