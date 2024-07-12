using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NetML.ML;

[SkipLocalsInit]
public sealed unsafe class Vector: IDisposable, IEnumerable<float> {
    public string name { get; }
    public int length { get; }
    public float* data { get; }

    private int allocated;

    public Vector(string name, int length, float* data) {
        this.name = name;
        this.length = length;
        this.data = data;
        allocated = 0;
    }

    public Vector(string name, int length) {
        if(length % 2 != 0) throw new ArgumentException($"length must be divisible by 2: {length}");

        this.name   = $"{name}[len={length}]";
        this.length = length;
        this.data   = (float*)NativeMemory.AlignedAlloc((UIntPtr)(length * sizeof(float)), 16);
        this.allocated = 1;

        // Console.WriteLine($"allocating vector {name}");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void load(ReadOnlySpan<float> data) {
        if(data.Length != length) throw new IndexOutOfRangeException($"{data.Length} != {length}");
        data.CopyTo(as_span());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void clear() => as_span().Clear();

    public float this[int i] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get {
            if((uint)i >= length) throw new IndexOutOfRangeException($"{i} >= {length}");
            return data[i];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            if((uint)i >= length) throw new IndexOutOfRangeException($"{i} >= {length}");
            data[i] = value;
        }
    }

    public static void multiply_elementwise(Vector left, Vector right, Vector result) {
        if(left.length != right.length) throw new Exception($"vectors must be same size: {left.name} != {right.name}");

        var left_ptr = left.data;
        var right_ptr = right.data;
        var result_ptr = result.data;

        for (var i = 0; i < left.length; i += 4) {
            var l = Vector128.LoadAligned(left_ptr + i);
            var r = Vector128.LoadAligned(right_ptr + i);
            var m = l * r;
            m.StoreAligned(result_ptr + i);
        }
    }

    public void add_elementwise(Vector other) {
        // Console.WriteLine($"{name} += {other.name}");
        var length = this.length;

        if(length != other.length) throw new Exception($"vectors must be same size: {name} != {other.name}");

        var data_ptr = this.data;
        var other_ptr = other.data;

        int i;

        for (i = 0; i < length - 4; i += 4) {
            var v = Vector128.LoadAligned(data_ptr + i);
            var o = Vector128.LoadAligned(other_ptr + i);
            var m = v + o;
            m.StoreAligned(data_ptr + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.LoadAligned(data_ptr + i);
            var o = Vector64.LoadAligned(other_ptr + i);
            var m = v + o;
            m.StoreAligned(data_ptr + i);
        }
    }

    public void add_elementwise_weighted(Vector other, float weight) {
        var length = this.length;

        if(length != other.length) throw new Exception($"vectors must be same size: {name} != {other.name}");

        var data_ptr = this.data;
        var other_ptr = other.data;

        int i;

        for (i = 0; i < length - 4; i += 4) {
            var v = Vector128.LoadAligned(data_ptr + i);
            var o = Vector128.LoadAligned(other_ptr + i);
            var m = v + o * weight;
            m.StoreAligned(data_ptr + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < length; i += 2) {
            var v = Vector64.LoadAligned(data_ptr + i);
            var o = Vector64.LoadAligned(other_ptr + i);
            var m = v + o * weight;
            m.StoreAligned(data_ptr + i);
        }
    }

    public static void add_elementwise(Vector left, Vector right, Vector result) {
        if(left.length != right.length) throw new Exception($"vectors must be same size: {left.name} != {right.name}");

        int i;

        for (i = 0; i < left.length - 4; i += 4) {
            var l = Vector128.LoadAligned(left.data + i);
            var r = Vector128.LoadAligned(right.data + i);
            var m = l + r;
            m.StoreAligned(result.data + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < left.length; i += 2) {
            var l = Vector64.LoadAligned(left.data + i);
            var r = Vector64.LoadAligned(right.data + i);
            var m = l + r;
            m.StoreAligned(result.data + i);
        }
    }

    public static void subtract_elementwise(Vector left, Vector right, Vector result) {
        if(left.length != right.length) throw new Exception($"vectors must be same size: {left.name} != {right.name}");

        int i;

        for (i = 0; i < left.length - 4; i += 4) {
            var l = Vector128.LoadAligned(left.data + i);
            var r = Vector128.LoadAligned(right.data + i);
            var m = l - r;
            m.StoreAligned(result.data + i);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < left.length; i += 2) {
            var l = Vector64.LoadAligned(left.data + i);
            var r = Vector64.LoadAligned(right.data + i);
            var m = l - r;
            m.StoreAligned(result.data + i);
        }
    }

    public static void dot(Matrix left, Vector right, Vector result) {
        if(left.output_count != right.length) throw new Exception($"{left.name} != {right.name}");
        if(left.input_count != result.length) throw new Exception($"{left.name} != {right.name}");

        var left_ptr  = left.data;
        var right_ptr = right.data;
        var result_ptr = result.data;

        var right_length = right.length;
        var result_length = result.length;

        for (var i = 0; i < right_length; i++) {
            var right_value = right_ptr[i];
            var sum = Vector128<float>.Zero;

            for (var j = 0; j < result_length; j += 4) {
                var l = Vector128.Load(left_ptr + i * result_length + j);
                sum += l * right_value;
            }

            result_ptr[i] = Vector128.Sum(sum);
        }
    }

    public static float mean_squared_error(Vector left, Vector right) {
        if(left.length != right.length) throw new Exception($"vectors must be same size: {left.name} != {right.name}");

        float sum_squared_errors = 0;

        int i;

        var left_length = left.length;

        for (i = 0; i < left_length - 4; i += 4) {
            var l = Vector128.LoadAligned(left.data + i);
            var r = Vector128.LoadAligned(right.data + i);
            var e = l - r;
            sum_squared_errors += Vector128.Sum(e);
        }

        // remaining elements if length is not a multiple of vectorSize
        for (; i < left_length; i += 2) {
            var l = Vector64.LoadAligned(left.data + i);
            var r = Vector64.LoadAligned(right.data + i);
            var e = l - r;
            sum_squared_errors += Vector64.Sum(e);
        }

        return sum_squared_errors / left_length;
    }

    public override string ToString()
        => $"{name} [{string.Join(", ", this.Select(static f => f))}]";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> as_span() => new Span<float>(data, length);

    private void ReleaseUnmanagedResources() {
        if (Interlocked.CompareExchange(ref allocated, 0, 1) != 1) return;
        Console.WriteLine($"releasing vector {name}");
        NativeMemory.AlignedFree(data);
    }

    public void Dispose() {
        if (allocated == 1) {
            ReleaseUnmanagedResources();
        }
        GC.SuppressFinalize(this);
    }

    public IEnumerator<float> GetEnumerator() {
        for (var i = 0; i < length; ++i) {
            yield return this[i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    ~Vector() {
        if (allocated == 1) {
            ReleaseUnmanagedResources();
        }
    }
}