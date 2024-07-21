using System.Runtime.CompilerServices;

namespace NetML.ML;

public readonly unsafe struct NativeSpan<T> where T: unmanaged {
    public readonly T* data;
    public readonly int length;

    public NativeSpan(T* data, int length) {
        this.data = data;
        this.length = length;
    }

    public NativeSpan(Span<T> span): this((T*)Unsafe.AsPointer(ref span[0]), span.Length) {
    }

    public T this[int index] {
        get {
            if ((uint)index >= length) throw new IndexOutOfRangeException();
            return data[index];
        }
        set {
            if ((uint)index >= length) throw new IndexOutOfRangeException();
            data[index] = value;
        }
    }

    public static implicit operator NativeSpan<T>(Span<T> span) => new(span);
    public static implicit operator T*(NativeSpan<T> span) => span.data;
}

public class Threaded {
    public static void run<T, T1>(Action<NativeSpan<T>, T1> function,
                                      Span<T> span,
                                      T1 param1,
                                      int threadCount
    ) where T: unmanaged {
        var length_per_thread = span.Length / threadCount;
        var remainder         = span.Length % threadCount;

        var countdown = new CountdownEvent(threadCount);

        for (var i = 0; i < threadCount; i++) {
            var start  = i * length_per_thread;
            var length = (i == threadCount - 1) ? length_per_thread + remainder : length_per_thread;

            NativeSpan<T> thread_span = span.Slice(start, length);

            ThreadPool.QueueUserWorkItem(
                                         _ => {
                                             try {
                                                 function(thread_span, param1);
                                             } finally {
                                                 countdown.Signal();
                                             }
                                         }
                                        );
        }

        countdown.Wait();
    }

    public static void run<T, T1, T2>(Action<NativeSpan<T>, T1, T2> function,
                                      Span<T> span,
                                      T1 param1,
                                      T2 param2,
                                      int threadCount
    ) where T: unmanaged {
        var length_per_thread = span.Length / threadCount;
        var remainder       = span.Length % threadCount;

        var countdown = new CountdownEvent(threadCount);

        for (var i = 0; i < threadCount; i++) {
            var start  = i * length_per_thread;
            var length = (i == threadCount - 1) ? length_per_thread + remainder : length_per_thread;

            NativeSpan<T> thread_span = span.Slice(start, length);

            ThreadPool.QueueUserWorkItem(
                                         _ => {
                                             try {
                                                 function(thread_span, param1, param2);
                                             } finally {
                                                 countdown.Signal();
                                             }
                                         }
                                        );
        }

        countdown.Wait();
    }
}