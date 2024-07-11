using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NetML.ML;

[SkipLocalsInit]
public sealed unsafe class Vector: IDisposable, IEnumerable<float> {
    public string name { get; }
    public int length { get; }

    private readonly float* data;
    private int allocated;

    public Vector(string name, int length) {
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

        // for (var i = 0; i < left.length; i++) {
        //    result[i] = left[i] * right[i];
        //}

        for (var i = 0; i < left.length; i += 4) {
            var l = Vector128.LoadAligned(left.data + i);
            var r = Vector128.LoadAligned(right.data + i);
            var m = l * r;
            m.StoreAligned(result.data + i);
        }
    }

    public void add_elementwise(Vector other) {
        if(length != other.length) throw new Exception($"vectors must be same size: {name} != {other.name}");

        for (var i = 0; i < length; i += 4) {
            var v = Vector128.LoadAligned(data + i);
            var o = Vector128.LoadAligned(other.data + i);
            var m = v + o;
            m.StoreAligned(data + i);
        }
    }

    public void add_elementwise_weighted(Vector other, float weight) {
        if(length != other.length) throw new Exception($"vectors must be same size: {name} != {other.name}");

        for (var i = 0; i < length; i += 4) {
            var v = Vector128.LoadAligned(data + i);
            var o = Vector128.LoadAligned(other.data + i);
            var m = v + o * weight;
            m.StoreAligned(data + i);
        }
    }


    public static void add_elementwise(Vector left, Vector right, Vector result) {
        if(left.length != right.length) throw new Exception($"vectors must be same size: {left.name} != {right.name}");

        for (var i = 0; i < left.length; i += 4) {
            var l = Vector128.LoadAligned(left.data + i);
            var r = Vector128.LoadAligned(right.data + i);
            var m = l + r;
            m.StoreAligned(result.data + i);
        }
    }

    public static void subtract_elementwise(Vector left, Vector right, Vector result) {
        if(left.length != right.length) throw new Exception($"vectors must be same size: {left.name} != {right.name}");

        for (var i = 0; i < left.length; i += 4) {
            var l = Vector128.LoadAligned(left.data + i);
            var r = Vector128.LoadAligned(right.data + i);
            var m = l - r;
            m.StoreAligned(result.data + i);
        }
    }

    public static void dot(Matrix left, Vector right, Vector result) {
        if(left.output_count != right.length) throw new Exception($"{left.name} != {right.name}");
        if(left.input_count != result.length) throw new Exception($"{left.name} != {right.name}");

        var left_ptr  = left.get_pointer();
        var right_ptr = right.get_pointer();
        var result_ptr = result.get_pointer();

        for (var i = 0; i < right.length; i++) {
            var right_value = right_ptr[i];
            var sum = Vector128<float>.Zero;

            for (var j = 0; j < result.length; j += 4) {
                var l = Vector128.LoadAligned(left_ptr + i * result.length + j);
                sum += l * right_value;
            }

            result_ptr[i] = Vector128.Sum(sum);
        }
    }

    public override string ToString()
        => $"{name} [{string.Join(", ", this.Select(static f => f))}]";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> as_span() => new Span<float>(data, length);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* get_pointer() => data;

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