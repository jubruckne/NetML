using System.Collections;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NetML.ML2;

public class Context: IDisposable {
    private readonly Dictionary<string, (object obj, nuint size, IntPtr memory)> allocations;

    public Context() {
        this.allocations = [];
    }

    public nuint allocated_memory {
        get {
            nuint size = 0;
            foreach (var a in allocations.Values) {
                size += a.size;
            }
            return size;
        }
    }

    public unsafe Tensor<T> allocate_tensor<T>(string name, IList data)
        where T: unmanaged, INumber<T> {

        var config = TensorExtensions.calculate_strides([data.Count]);
        var array  = NativeMemory.AlignedAlloc((UIntPtr)(config.linear_length * sizeof(float)), 16);
        var tensor = Tensor<T>.create(name, (T*)array, config.linear_length, config.shape, config.strides);
        allocations.Add(name, (tensor, (nuint)config.linear_length * sizeof(float), (IntPtr)array));

        tensor.clear();
        tensor.print();

        for (var i = 0; i < data.Count; ++i) {
            tensor[[i]] = (T)Convert.ChangeType(data[i], typeof(T))!;
        }

        return tensor;
    }


    public unsafe Tensor<T> allocate_tensor<T>(string name, ReadOnlySpan<int> shape)
        where T: unmanaged, INumber<T> {
        var config = TensorExtensions.calculate_strides(shape);
        var array  = NativeMemory.AlignedAlloc((UIntPtr)(config.linear_length * sizeof(float)), 16);
        var tensor = Tensor<T>.create(name, (T*)array, config.linear_length, config.shape, config.strides);
        allocations.Add(name, (tensor, (nuint)config.linear_length * sizeof(float), (IntPtr)array));
        return tensor;
    }

    public T get_tensor<T>(string name) => (T)allocations[name].obj;

    public unsafe void release() {
        foreach (var a in allocations.Values) {
            Console.WriteLine($"releasing {a.obj}...");
            NativeMemory.AlignedFree((void*)a.memory);
        }
        allocations.Clear();
    }

    void IDisposable.Dispose() {
        release();
        GC.SuppressFinalize(this);
    }
}