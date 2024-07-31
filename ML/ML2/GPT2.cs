using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.FlowAnalysis;

namespace NetML.ML2;

public class GPT2 {
    private readonly int vocab_size; // 50257: Vocabulary size
    private readonly int n_layer;    // 12: Number of layers (blocks)
    private readonly int n_head;     // 12: Number of attention heads
    private readonly int n_embed;    // 768: Embedding dimension
    private readonly int n_ctx;      // 1024: Context size (max sequence length), also block size

    public class State {
        /// <summary>
        /// tokenized input
        /// </summary>
        public Tensor<int> input;

        /// <summary>
        /// [n_ctx, n_embed]
        ///
        /// activation, at current time stamp
        /// </summary>
        public Tensor<float> x;

        /// <summary>
        /// [n_ctx, n_embed]
        /// activation, inside a residual stream
        /// </summary>
        public Tensor<float> xb;

        /// <summary>
        /// [layer, n_ctx, n_embed, n_embed * 3]
        /// query, key and values
        /// </summary>
        public Tensor<float> qkv;
    }

    // public float[] xb2;    // an additional buffer just for convenience (dim,)
    // public float[] hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    // public float[] hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    // public float[] q;      // query (dim,)
    // public float[] k;      // key (dim,)
    // public float[] v;      // value (dim,)
    // public float[] att;    // buffer for scores/attention values (n_heads, seq_len)
    // public float[] logits; // output logits

    private readonly GGUF.Tokenizer tokenizer;
    private readonly State state;

    private readonly List<Block> layers; // Transformer blocks

    public class Block {
        public readonly Tensor<float> qkv;      // Shape: [3, embedDim, embedDim]
        public readonly Tensor<float> qkv_bias; // Shape: [3 * embedDim]

        public readonly Tensor<float> attn_proj;      // Shape: [embedDim, embedDim]
        public readonly Tensor<float> attn_proj_bias; // Shape: [embedDim]

        public readonly Tensor<float> mlp1;      // Shape: [embedDim, 4 * embedDim]
        public readonly Tensor<float> mlp1_bias; // Shape: [4 * embedDim]

        public readonly Tensor<float> mlp2;      // Shape: [4 * embedDim, embedDim]
        public readonly Tensor<float> mlp2_bias; // Shape: [embedDim]

        public readonly Tensor<float> ln1; // Shape: [embedDim] (gamma)
        public readonly Tensor<float> ln1_bias; // Shape: [embedDim] (beta)

        public readonly Tensor<float> ln2; // Shape: [embedDim] (gamma)
        public readonly Tensor<float> ln2_bias; // Shape: [embedDim] (beta)

        public Block(Tensor<float> qkv,
                     Tensor<float> qkv_bias,
                     Tensor<float> attn_proj,
                     Tensor<float> attn_proj_bias,
                     Tensor<float> mlp1,
                     Tensor<float> mlp1_bias,
                     Tensor<float> mlp2,
                     Tensor<float> mlp2_bias,
                     Tensor<float> ln1,
                     Tensor<float> ln1_bias,
                     Tensor<float> ln2,
                     Tensor<float> ln2_bias
        ) {
            this.qkv            = qkv;
            this.qkv_bias       = qkv_bias;
            this.attn_proj      = attn_proj;
            this.attn_proj_bias = attn_proj_bias;
            this.mlp1           = mlp1;
            this.mlp1_bias      = mlp1_bias;
            this.mlp2           = mlp2;
            this.mlp2_bias      = mlp2_bias;
            this.ln1            = ln1;
            this.ln1_bias       = ln1_bias;
            this.ln2            = ln2;
            this.ln2_bias       = ln2_bias;
        }
    }

    private class Tensors {
        /// <summary>
        /// [vocab_size, n_embed]
        /// </summary>
        public Tensor<float> token_embd;

        /// <summary>
        /// [context_length, n_embed]
        /// </summary>
        public Tensor<float> position_embd;

        /// <summary>
        /// [layer, n_embed]
        /// </summary>
        public Tensor<float> attn_norm_weight;

        /// <summary>
        /// [layer, n_embed]
        /// </summary>
        public Tensor<float> attn_norm_bias;

        /// <summary>
        /// [layer, n_embed, n_embed * 3]
        /// </summary>
        public Tensor<float> attn_qkv_weight;

        /// <summary>
        /// [layer, n_embed * 3]
        /// </summary>
        public Tensor<float> attn_qkv_bias;

        /// <summary>
        /// [n_embed] (gamma)
        /// </summary>
        public Tensor<float> ln_f;

        /// <summary>
        /// [n_embed] (beta)
        /// </summary>
        public Tensor<float> ln_f_bias;

    }

    private readonly Tensors tensors = new();

    public GPT2(Context ctx, GGUF.File file) {
        vocab_size = file.vocab_size;
        Console.WriteLine($"vocab_size: {vocab_size}");

        n_layer = file.metadata.get<int>("gpt2.block_count");
        Console.WriteLine($"n_layer: {n_layer}");

        n_head = file.metadata.get<int>("gpt2.attention.head_count");
        Console.WriteLine($"n_head: {n_head}");

        n_embed = file.metadata.get<int>("gpt2.embedding_length");
        Console.WriteLine($"n_embd: {n_embed}");

        n_ctx = file.metadata.get<int>("gpt2.context_length");
        Console.WriteLine($"context_length: {n_ctx}");

        load_tensors(ctx, file);

        state = new() {
                          input = ctx.allocate_tensor<int>("input", [n_ctx]),
                          x     = ctx.allocate_tensor<float>("x", [n_ctx, n_embed]),
                          xb    = ctx.allocate_tensor<float>("xb", [n_ctx, n_embed]),
                          qkv   = ctx.allocate_tensor<float>("qkv", [n_ctx, n_embed, n_embed * 3]),
                      };

        Console.WriteLine();

        tokenizer = file.build_tokenizer();
        Console.WriteLine("\n******************\n");
        Console.WriteLine(generate("My name is", 50));
        Console.WriteLine();
    }

    public string generate(string prompt, int max_new_tokens) {
        var context = tokenizer.tokenize_and_convert_to_ids(prompt);

        while (context.Count < max_new_tokens) {
            var logits = forward(context);

            var nextToken = sample_from_logits(logits[^1]);
            context.Add(nextToken);

            if (nextToken == 50256) // End of text token
                break;
        }

        return tokenizer.decode(context);

        Tensor<float> forward(List<int> ids) {
            state.input.clear();
            state.input.insert(ids.ToArray());

            // Embedding
            state.x.token_embedding(tensors.token_embd, state.input);
            state.x.positional_encoding(tensors.position_embd);

            foreach (var l in layers) {
                state.x = forward_layer(l, state.x);
            }

            state.x.layer_norm(tensors.ln_f, tensors.ln_f_bias);
            return state.x;
        }

        Tensor<float> forward_layer(Block block, Tensor<float> x) {
            // Input shape:  [n_ctx, n_embed]
            // Output shape: [n_ctx, n_embed]

            var a = attention(x.layer_norm(block.ln1, block.ln1_bias));
            x = x.add(a);

            var m = mlp(x.layer_norm(block.ln2, block.ln2_bias));
            x = x.add(m);

            return x;
        }

        Tensor<float> attention(Tensor<float> x) {
            // This method performs multi-head attention
            // Input shape: [n_ctx, n_embd]
            // Output shape: [n_ctx, n_embd]

            var qkv = state.qkv.slice([layer]);

            var qkv = apply_qkv(x); // Shape: [3, n_ctx, n_head, n_embd / n_head]
            var a   = MultiHeadAttention(qkv[0], qkv[1], qkv[2]);
            return ApplyLinearWithBias(a, c_proj, c_proj_bias);

            return x;
        }

        Tensor<float> apply_qkv(Tensor<float> x) {
            // Apply the combined QKV projection
            // Input shape: [n_ctx, n_embd]
            // Output shape: [3, n_ctx, n_head, n_embd / n_head]


            qkv.linear_projection(xb, tensors.attn_qkv_weight.slice([layer]), tensors.attn_qkv_bias.slice([layer]));


            var result = new float[3][][];
            for (int i = 0; i < 3; i++)
            {
                result[i] = new float[n_ctx][];
                for (int j = 0; j < n_ctx; j++)
                {
                    result[i][j] = new float[n_embd];
                    for (int k = 0; k < n_embd; k++)
                    {
                        result[i][j][k] = Dot(x[j], c_attn[i, .., k]) + c_attn_bias[i * n_embd + k];
                    }
                }
            }
            return result;
        }

        Tensor<float> mlp(Tensor<float> x) {
            return x;
        }


    }

    /*input.clear();
    input.insert(ids.ToArray());

    // Embedding
    x.token_embedding(tensors.token_embd, input);
    x.positional_encoding(tensors.position_embd);

    // forward all the layers
    for (var layer = 0; layer < n_layer; layer++) {
        Console.WriteLine($"layer: {layer}");
        // attention norm
        xb.layer_norm(x, tensors.attn_norm_weight.slice([layer]), tensors.attn_norm_bias.slice([layer]));
        qkv.linear_projection(xb, tensors.attn_qkv_weight.slice([layer]), tensors.attn_qkv_bias.slice([layer]));

        var (q, k, v) = qkv.split("q", "k", "v");
        var qH = q.split(n_head);
        var kH = k.split(n_head);
        var vH = v.split(n_head);

        for (var h = 0; h < n_head; h++) {
            var keys    = kH[h];
            var queries = qH[h];
            var values  = vH[h];

            keys.multiply(queries);
            keys.multiply(values);
        }

        q.print();

    }
}*/

    private void load_tensors(Context ctx, GGUF.File file) {
        tensors.token_embd    = ctx.allocate_tensor<float>("wte", [vocab_size, n_embed]);
        file.load_tensor_data("token_embd.weight", tensors.token_embd, true);

        tensors.position_embd = ctx.allocate_tensor<float>("wpe", [n_ctx, n_embed]);
        file.load_tensor_data("position_embd.weight", tensors.position_embd, true);

        tensors.attn_norm_weight = ctx.allocate_tensor<float>("attn_norm.weight", [n_layer, n_embed]);
        tensors.attn_norm_bias = ctx.allocate_tensor<float>("attn_norm.bias", [n_layer, n_embed]);

        tensors.attn_qkv_weight = ctx.allocate_tensor<float>("attn_qkv.weight", [n_layer, n_embed, n_embed * 3]);
        tensors.attn_qkv_bias = ctx.allocate_tensor<float>("attn_qkv.bias", [n_layer, n_embed * 3]);

        for (var l = 0; l < n_layer; l++) {
            Console.WriteLine($"layer {l}...");
            // blk.0.attn_norm.weight
            file.load_tensor_data($"blk.{l}.attn_norm.weight", tensors.attn_norm_weight.slice([l]));

            // blk.0.attn_norm.bias
            file.load_tensor_data($"blk.{l}.attn_norm.bias", tensors.attn_norm_bias.slice([l]));

            // blk.0.attn_norm.weight
            file.load_tensor_data($"blk.{l}.attn_qkv.weight", tensors.attn_qkv_weight.slice([l]));

            // blk.0.attn_qkv.bias
            file.load_tensor_data($"blk.{l}.attn_qkv.bias", tensors.attn_qkv_bias.slice([l]));
        }

        tensors.ln_f = ctx.allocate_tensor<float>("output_norm.weight", [n_embed]);
        tensors.ln_f_bias = ctx.allocate_tensor<float>("output.norm_bias", [n_embed]);

        file.load_tensor_data("output_norm.weight", tensors.ln_f);
        file.load_tensor_data("output_norm.bias", tensors.ln_f_bias);

        Console.WriteLine($"All tensors loaded. Allocated: {ctx.allocated_memory:N0}\n");
    }
}