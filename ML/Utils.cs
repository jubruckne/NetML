using System.Collections;
using System.Diagnostics.CodeAnalysis;

namespace NetML.ML2;

public static class Utils {
    public static string join<T>(this IEnumerable<T> values, string delimiter = ", ")
        => string.Join(delimiter, values);

    public static T create_instance<T>(params object[] args)
        => (T)Activator.CreateInstance(typeof(T), args)!;
}

public class Bag: IEnumerable<object> {
    protected readonly Dictionary<string, object> data;

    public Bag()
        => data = new Dictionary<string, object>();

    public void add<T>(string key, T item)
        where T: notnull
        => data.Add(key, item);

    public virtual object this[string key] => data[key];
    public int length => data.Count;

    public bool try_get_value<T>(string key, [MaybeNullWhen(false)] out T value) {
        if (data.TryGetValue(key, out var v)) {
            value = (T)v;
            return true;
        }

        value = default;
        return false;
    }

    public IEnumerator<object> GetEnumerator() {
        foreach (var kv in data) {
            yield return kv.Value;
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public IEnumerable<string> keys => data.Keys;
    public IEnumerable<object> values => data.Values;
    public IEnumerable<(string key, object value)> keys_and_values {
        get {
            foreach (var kv in data) {
                yield return (kv.Key, kv.Value);
            }
        }
    }

    public T get<T>(string key) {
        if (data.TryGetValue(key, out var v)) {
            if (v is T value)
                return value;

            if (v is uint v_u && typeof(T) == typeof(int)) {
                int v_int = (int)v_u;
                return (T)(object)v_int;
            }

            Console.WriteLine($"Key {key}:{v.GetType().Name} is not of type {typeof(T).Name}");


            return (T)v;

            throw new Exception($"Key {key}:{v.GetType().Name} is not of type {typeof(T).Name}");
        }

        throw new KeyNotFoundException(key);
    }
}

public class Bag<T>: Bag, IEnumerable<T> {
    public new T this[string key] => (T)base[key];

    public new IEnumerator<T> GetEnumerator() {
        foreach (var kv in data) {
            yield return (T)kv.Value;
        }
    }

    public new IEnumerable<T> values => data.Values.Cast<T>();

    public new IEnumerable<(string key, T value)> keys_and_values {
        get {
            foreach (var kv in data) {
                yield return (kv.Key, (T)kv.Value);
            }
        }
    }

}