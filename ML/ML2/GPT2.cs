namespace NetML.ML2;

public class GPT2 {
    private readonly int vocab_size;     // 50257
    private readonly int n_layer;        // 12
    private readonly int n_head;         // 12
    private readonly int n_embd;         // 768
    private readonly int context_length; // 1024 also block size

    private readonly Tensor<float> wte;
    private readonly Tensor<float> wpe;
    private readonly Tensor<float> input;
    private readonly Tensor<float> x;

    public readonly Graph graph;

    public GPT2(Context ctx, GGUF.File file) {
        vocab_size = file.vocab_size;
        Console.WriteLine($"vocab_size: {vocab_size}");

        n_layer = file.metadata.get<int>("gpt2.block_count");
        Console.WriteLine($"n_layer: {n_layer}");

        n_head = file.metadata.get<int>("gpt2.attention.head_count");
        Console.WriteLine($"n_head: {n_head}");

        n_embd = file.metadata.get<int>("gpt2.embedding_length");
        Console.WriteLine($"n_embd: {n_embd}");

        context_length = file.metadata.get<int>("gpt2.context_length");
        Console.WriteLine($"context_length: {context_length}");

        wte = ctx.allocate_tensor<float>("wte", [vocab_size, n_embd]);
        wpe = ctx.allocate_tensor<float>("wpe", [context_length, n_embd]);

        file.load_tensor_data("token_embd.weight", wte, true);
        file.load_tensor_data("position_embd.weight", wpe, true);

        input = ctx.allocate_tensor<float>("input", [context_length]);
        x = ctx.allocate_tensor<float>("x", [context_length, n_embd]);

        Console.WriteLine();

        graph = Graph.with_target(x);
        graph.embedding(x, wte, input);

    }
}