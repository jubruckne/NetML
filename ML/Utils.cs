using System.Collections;
using System.Diagnostics.CodeAnalysis;

namespace NetML.ML2;

public static class Utils {
    public static string join<T>(this IEnumerable<T> values, string delimiter = ", ")
        => string.Join(delimiter, values);

    public static string join<T>(this ReadOnlySpan<T> values, string delimiter = ", ")
        => string.Join(delimiter, values.ToArray());



    public static T create_instance<T>(params object[] args)
        => (T)Activator.CreateInstance(typeof(T), args)!;
}