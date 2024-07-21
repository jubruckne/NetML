using System.Diagnostics;
using System.Text;

namespace NetML.ML;

public static class Metrics {
    public static readonly Stopwatch Vector_Insert = new Stopwatch();
    public static readonly Stopwatch Matrix_Multiply = new Stopwatch();
    public static readonly Stopwatch Vector_Add_Elementwise = new Stopwatch();
    public static readonly Stopwatch Activation_Sigmoid = new Stopwatch();

    public static void print() {
        StringBuilder sb = new StringBuilder();

        sb.AppendLine($"Vector_Insert: {Vector_Insert.ElapsedMilliseconds}");
        sb.AppendLine($"Matrix_Multiply: {Matrix_Multiply.ElapsedMilliseconds}");
        sb.AppendLine($"Vector_Add_Elementwise: {Vector_Add_Elementwise.ElapsedMilliseconds}");
        sb.AppendLine($"Activation_Sigmoid: {Activation_Sigmoid.ElapsedMilliseconds}");

        Console.WriteLine();
        Console.WriteLine(sb.ToString());
        Console.WriteLine();
    }
}