using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Text;
using System.Text.Json.Serialization;

namespace NetML.ML;

[JsonConverter(typeof(VectorConverter))]
[SkipLocalsInit]
public sealed unsafe class Vector: IDisposable,  ITensor<float>, IEnumerable<float> {
    public string name { get; }
    public int[] shape { get; }
    public int length { get; }
    public float* data { get; }

    public static bool use_accelerate = false;

    int ITensor<float>.linear_length => length;

    private int allocated;

    public Vector(string name, int length, float* data) {
        if (length % 2 != 0) throw new ArgumentException($"length must be divisible by 2: {length}");

        this.name   = name;
        this.length = length;
        this.shape = [length];
        this.data   = data;
        this.allocated = 0;
    }

    public Vector(string name, int length) {
        if (length % 2 != 0) throw new ArgumentException($"length must be divisible by 2: {length}");

        this.name   = name;
        this.length = length;
        this.shape = [length];
        this.data = (float*)NativeMemory.AlignedAlloc((UIntPtr)(length * sizeof(float)), 16);
        this.allocated   = 1;
    }

    public void insert(ReadOnlySpan<float> array) {
        if (array.Length != length) throw new IndexOutOfRangeException($"{array.Length} != {length}");
        array.CopyTo(as_span());
    }

    public void insert(Vector other) {
        if (length != other.length) throw new IndexOutOfRangeException($"{length} != {other.length}");
        other.as_readonly_span().CopyTo(as_span());
    }

    public void clear() {
        as_span().Clear();
    }

    float ITensor<float>.this[ReadOnlySpan<int> indices] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this[indices[0]];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => this[indices[0]] = value;
    }

    public float this[params int[] indices] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this[indices[0]];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => this[indices[0]] = value;
    }

    public float this[int i] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get {
            if ((uint)i >= length) throw new IndexOutOfRangeException($"{i} >= {length}");
            return data[i];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            if ((uint)i >= length) throw new IndexOutOfRangeException($"{i} >= {length}");
            data[i] = value;
        }
    }

    public static void multiply_elementwise(Vector left, Vector right, Vector result) {
        var left_length = left.length;

        if (left_length != right.length) throw new($"vectors must be same size: {left.name} != {right.name}");

        if (use_accelerate) {
            Accelerate.vDSP_vmul(left.data, 1, right.data, 1, result.data, 1, (uint)left.length);
        } else {
            var left_ptr   = left.data;
            var right_ptr  = right.data;
            var result_ptr = result.data;

            for (var i = 0; i < left_length; i += 4) {
                var l = Vector128.Load(left_ptr + i);
                var r = Vector128.Load(right_ptr + i);
                var m = l * r;
                m.Store(result_ptr + i);
            }
        }
    }

    public void add_elementwise(Vector other) {
        // Console.WriteLine($"{name} += {other.name}");
        var length = this.length;

        if (length != other.length) throw new($"vectors must be same size: {name} != {other.name}");

        if (use_accelerate) {
            Accelerate.vDSP_vadd(data, 1, other.data, 1, data, 1, (uint)length);
        } else {
            var data_ptr  = data;
            var other_ptr = other.data;

            int i;

            for (i = 0; i < length - 4; i += 4) {
                var v = Vector128.Load(data_ptr + i);
                var o = Vector128.Load(other_ptr + i);
                var m = v + o;
                m.Store(data_ptr + i);
            }

            // remaining elements if length is not a multiple of vectorSize
            for (; i < length; i += 2) {
                var v = Vector64.Load(data_ptr + i);
                var o = Vector64.Load(other_ptr + i);
                var m = v + o;
                m.Store(data_ptr + i);
            }
        }
    }

    public void add_elementwise_weighted(Vector other, float weight) {
        var length = this.length;

        if (length != other.length) throw new($"vectors must be same size: {name} != {other.name}");

        var data_ptr  = data;
        var other_ptr = other.data;

        int i;

        for (i = 0; i < length - 4; i += 4) {
            AdvSimd.LoadVector128(data);
            var v = Vector128.Load(data_ptr + i);
            var o = Vector128.Load(other_ptr + i);
            var m = v + o * weight;
            m.Store(data_ptr + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.Load(data_ptr + i);
            var o = Vector64.Load(other_ptr + i);
            var m = v + o * weight;
            m.Store(data_ptr + i);
        }
    }

    public static void add_elementwise(Vector left, Vector right, Vector result) {
        if (left.length != right.length) throw new($"vectors must be same size: {left.name} != {right.name}");

        int i;

        for (i = 0; i < left.length - 4; i += 4) {
            var l = Vector128.Load(left.data + i);
            var r = Vector128.Load(right.data + i);
            var m = l + r;
            m.Store(result.data + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < left.length; i += 2) {
            var l = Vector64.Load(left.data + i);
            var r = Vector64.Load(right.data + i);
            var m = l + r;
            m.Store(result.data + i);
        }
    }

    public static void subtract_elementwise(Vector left, Vector right, Vector result) {
        var left_length = left.length;

        if (left_length != right.length) throw new($"vectors must be same size: {left.name} != {right.name}");
        if (left_length != result.length) throw new($"vectors must be same size: {left.name} != {result.name}");

        if (use_accelerate) {
            Accelerate.vDSP_vsub(right.data, 1, left.data, 1, result.data, 1, (uint)left.length);
        } else {
            var left_ptr   = left.data;
            var right_ptr  = right.data;
            var result_ptr = result.data;

            int i;

            for (i = 0; i < left_length - 4; i += 4) {
                var l = Vector128.Load(left_ptr + i);
                var r = Vector128.Load(right_ptr + i);
                var m = l - r;
                m.Store(result_ptr + i);
            }

            // remaining elements if length is not a multiple of vectorSize
            for (; i < left_length; i += 2) {
                var l = Vector64.Load(left_ptr + i);
                var r = Vector64.Load(right_ptr + i);
                var m = l - r;
                m.Store(result_ptr + i);
            }
        }
    }

    public static void dot(Matrix left, Vector right, Vector result) {
        var right_length  = right.length;
        var result_length = result.length;

        if (left.output_count != right.length || left.input_count != result.length)
            throw new ArgumentException("Matrix and vector dimensions do not match");

        var left_ptr   = left.data;
        var right_ptr  = right.data;
        var result_ptr = result.data;

        if (use_accelerate) {
            const int CblasRowMajor = 101;
            const int CblasNoTrans  = 111;

            Accelerate.cblas_sgemv(
                                   CblasRowMajor, // Row-major order
                                   CblasNoTrans,  // No transpose
                                   result_length, // Number of rows in matrix
                                   right_length,  // Number of columns in matrix
                                   1.0f,          // Alpha scalar
                                   left_ptr,      // Matrix data
                                   right_length,  // Leading dimension of matrix
                                   right_ptr,     // Input vector
                                   1,             // Increment for input vector
                                   0.0f,          // Beta scalar
                                   result_ptr,    // Output vector
                                   1              // Increment for output vector
                                  );
        } else {
            for (var i = 0; i < right_length; i++) {
                var right_value = right_ptr[i];
                var sum         = Vector128<float>.Zero;

                for (var j = 0; j < result_length; j += 4) {
                    var l = Vector128.Load(left_ptr + i * result_length + j);
                    sum += l * right_value;
                }

                result_ptr[i] = Vector128.Sum(sum);
            }
        }
    }

    public float sum() {
        var length   = this.length;
        var data_ptr = data;

        int i;
        var sum = 0f;

        for (i = 0; i < length - 4; i += 4) {
            var v = Vector128.Load(data_ptr + i);
            sum += Vector128.Sum(v);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.Load(data_ptr + i);
            sum += Vector64.Sum(v);
        }

        return sum;
    }

    public static float mean_squared_error(Vector left, Vector right) {
        var left_length = left.length;

        if (left_length != right.length) throw new($"vectors must be same size: {left.name} != {right.name}");

        float sum_squared_errors = 0;

        int i;

        for (i = 0; i < left_length - 4; i += 4) {
            var l = Vector128.Load(left.data + i);
            var r = Vector128.Load(right.data + i);
            var e = l - r;
            sum_squared_errors += Vector128.Sum(e * e);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < left_length; i += 2) {
            var l = Vector64.Load(left.data + i);
            var r = Vector64.Load(right.data + i);
            var e = l - r;
            sum_squared_errors += Vector64.Sum(e * e);
        }

        return sum_squared_errors / left_length;
    }

    public int index_of(float value) {
        var length = this.length;

        var search_value = Vector128.Create(value);

        for (var i = 0; i <= length - 4; i += 4) {
            var v = Vector128.Load(data + i);

            var comp = Vector128.Equals(v, search_value);
            if (comp.Equals(Vector128<float>.Zero)) continue;

            // If there is a match, find the exact index
            for (var j = 0; j < Vector128<float>.Count; j++)
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (v.GetElement(j) == value)
                    return i + j;
        }

        return -1;
    }

    public int index_of_max_value() {
        if (length == 0) return -1;

        var max_index = 0;
        var max_value = this[0];

        for (var i = 1; i < length; ++i)
            if (this[i] > max_value) {
                max_value = this[i];
                max_index = i;
            }

        return max_index;
    }

    /*
    public static implicit operator Vector(float[] data) => new Vector("temp", data);
    public static implicit operator Vector(Span<float> data) => new Vector("temp", data.ToArray());
    public static implicit operator Vector(ReadOnlySpan<float> data) => new Vector("temp", data.ToArray());
*/
    public override string ToString() {
        return $"{name}[len={length}]";
    }

    public string print() {
        StringBuilder sb = new();

        sb.AppendLine(ToString());
        sb.Append("  ");
        for (var i = 0; i < length; ++i) {
            sb.Append($"{this[i]:N4}, ");
            if (i % 16 == 0) sb.Append("\n  ");
        }

        return sb.ToString().TrimEnd(',', ' ');
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> as_span()
        => new(data, length);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> as_readonly_span()
        => new(data, length);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool is_aligned()
        => (nuint)data % 16 == 0;

    public IEnumerator<float> GetEnumerator() {
        for (var i = 0; i < length; ++i) yield return this[i];
    }

    IEnumerator IEnumerable.GetEnumerator() {
        return GetEnumerator();
    }

    private void ReleaseUnmanagedResources() {
        var alloc = Interlocked.CompareExchange(ref allocated, 0, 1);

        if (alloc == 1)
            // Console.WriteLine($"releasing vector {name}");
            NativeMemory.AlignedFree(data);
    }

    public void Dispose() {
        if (allocated == 0) return;
        ReleaseUnmanagedResources();
        GC.SuppressFinalize(this);
    }

    ~Vector() {
        if (allocated == 0) return;
        ReleaseUnmanagedResources();
    }
}