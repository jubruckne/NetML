using System.Collections;
using System.Diagnostics.CodeAnalysis;

namespace NetML.ML2;

public sealed class Bag: Bag<object> {
    public Bag(): base() {}

    public bool try_get_value<T>(string key, [MaybeNullWhen(false)] out T value) {
        if (data.TryGetValue(key, out var v)) {
            if (v is T tt) {
                value = tt;
                return true;
            }
        }

        value = default;
        return false;
    }

    public T get<T>(string key) {
        if (data.TryGetValue(key, out var v)) {
            if (v is T value)
                return value;

            if (v is uint v_u && typeof(T) == typeof(int)) {
                var v_int = (int)v_u;
                return (T)(object)v_int;
            }

            Console.WriteLine($"Key {key}:{v.GetType().Name} is compatible with type {typeof(T).Name}");

            return (T)v;

        }

        throw new KeyNotFoundException(key);
    }
}

public class Bag<T>: IEnumerable<T> {
    public int length => data.Count;

    protected readonly Dictionary<string, T> data;

    public Bag()
        => data = new Dictionary<string, T>();

    public void add(string key, T item)
        => data.Add(key, item);

    public bool try_get_value(string key, [MaybeNullWhen(false)] out T value) {
        if (data.TryGetValue(key, out var v)) {
            value = v;
            return true;
        }

        value = default;
        return false;
    }

    public T get(string key) {
        if (data.TryGetValue(key, out var v)) {
            return v;
        }

        throw new KeyNotFoundException(key);
    }

    public T this[string key] => data[key];

    public IEnumerator<T> GetEnumerator() {
        foreach (var kv in data) {
            yield return kv.Value;
        }
    }

    public IEnumerable<string> keys => data.Keys;

    public IEnumerable<T> values => data.Values;

    public IEnumerable<(string key, T value)> keys_and_values {
        get {
            foreach (var kv in data) {
                yield return (kv.Key, kv.Value);
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}