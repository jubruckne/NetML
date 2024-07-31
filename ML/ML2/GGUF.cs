using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace NetML.ML2;

public static class GGUF {
    public static File from_file(string filePath) {
        var mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open);
        return new File(mmf);
    }

    public class File: IDisposable {
        public uint version { get; }
        public ulong tensor_count { get; }
        public ulong metadata_count { get; }
        public Bag metadata { get; }
        public Bag<TensorInfo> tensors { get; }
        public uint alignment { get; }

        private const uint GGUF_MAGIC = 0x46554747;
        private readonly MemoryMappedFile mmf;
        private readonly MemoryMappedViewAccessor mmf_accessor;

        public File(MemoryMappedFile mmf) {
            this.mmf = mmf;
            using var reader = new BinaryReader(mmf.CreateViewStream());

            var magic = reader.ReadUInt32();
            if (magic != GGUF_MAGIC) {
                throw new InvalidDataException("Not a valid GGUF file");
            }

            version = reader.ReadUInt32();
            if (version != 3) {
                throw new NotSupportedException($"Unsupported GGUF version: {version}");
            }

            tensor_count   = reader.ReadUInt64();
            metadata_count = reader.ReadUInt64();
            metadata       = read_metadata(reader, metadata_count);
            alignment      = get_meta_data_value<uint>("general.alignment", 32);
            tensors        = read_tensor_info(reader, tensor_count, alignment);

            mmf_accessor = mmf.CreateViewAccessor();
        }

        public int vocab_size {
            get {
                if(metadata.try_get_value("tokenizer.ggml.tokens", out List<object> v)) {
                    Console.WriteLine(v);
                    return v.Count;
                }

                return -1;
            }
        }

        private static Bag read_metadata(BinaryReader reader, ulong count) {
            var metadata = new Bag();
            for (ulong i = 0; i < count; i++) {
                var key   = read_string(reader);
                var value = read_value(reader);
                metadata.add(key, value);
            }

            return metadata;
        }

        private static Bag<TensorInfo> read_tensor_info(BinaryReader reader, ulong count, uint alignment) {
            var tensors = new Bag<TensorInfo>();
            for (ulong i = 0; i < count; i++) {
                var name   = read_string(reader);
                var shape  = read_uint64_array(reader);
                var type   = (GGMLType)reader.ReadUInt32();
                var offset = reader.ReadUInt64();
                var config = TensorExtensions.calculate_strides(shape);

                //Console.WriteLine($"**** read tensor info{name}: linear_length: {config.linear_length}, strides: [{config.strides.join()}], shape: [{shape.join()}]");

                var tensor = new TensorInfo {
                                                name          = name,
                                                shape         = config.shape,
                                                strides       = config.strides,
                                                type          = type,
                                                offset        = offset,
                                                linear_length = config.linear_length
                                            };
                tensors.add(tensor.name, tensor);
            }

            // Align to the specified alignment
            var currentPosition = reader.BaseStream.Position;
            var alignedPosition = (currentPosition + alignment - 1) / alignment * alignment;
            reader.BaseStream.Seek(alignedPosition, SeekOrigin.Begin);

            return tensors;
        }

        private static string read_string(BinaryReader reader) {
            var length      = reader.ReadUInt64();
            var stringBytes = reader.ReadBytes((int)length);
            return Encoding.UTF8.GetString(stringBytes);
        }

        private static object read_value(BinaryReader reader, MetadataValueType? type = null) {
            if (type is null) {
                type = (MetadataValueType)reader.ReadUInt32();
            }

            return type switch {
                MetadataValueType.UINT8 => reader.ReadByte(),
                MetadataValueType.INT8 => reader.ReadSByte(),
                MetadataValueType.UINT16 => reader.ReadUInt16(),
                MetadataValueType.INT16 => reader.ReadInt16(),
                MetadataValueType.UINT32 => reader.ReadUInt32(),
                MetadataValueType.INT32 => reader.ReadInt32(),
                MetadataValueType.FLOAT32 => reader.ReadSingle(),
                MetadataValueType.BOOL => reader.ReadBoolean(),
                MetadataValueType.STRING => read_string(reader),
                MetadataValueType.ARRAY => read_array(reader),
                MetadataValueType.UINT64 => reader.ReadUInt64(),
                MetadataValueType.INT64 => reader.ReadInt64(),
                MetadataValueType.FLOAT64 => reader.ReadDouble(),
                _ => throw new NotSupportedException($"Unsupported metadata value type: {type}")
            };
        }

        private static List<object> read_array(BinaryReader reader) {
            var elementType = (MetadataValueType)reader.ReadUInt32();
            var count       = reader.ReadUInt64();
            var array       = new List<object>();
            for (ulong i = 0; i < count; i++) {
                array.Add(read_value(reader, elementType));
            }

            return array;
        }

        private static int[] read_uint64_array(BinaryReader reader) {
            var count = reader.ReadUInt32();
            return Enumerable.Range(0, (int)count).Select(_ => (int)reader.ReadUInt64()).ToArray();
        }

        private T? get_meta_data_value<T>(string key, T? default_value)
            => metadata.try_get_value(key, out T? value) ? value : default_value;

        public unsafe NetML.ML2.Tensor<float> load_tensor(string tensor_name) {
            var tensor = tensors[tensor_name];

            return NetML.ML2.Tensor<float>.create(
                                                  tensor.name,
                                                  load_tensor_data<float>(tensor.name),
                                                  tensor.linear_length,
                                                  tensor.shape
                                                 );
        }

        public unsafe T* load_tensor_data<T>(string tensor_name)
            where T: unmanaged {
            var tensor = tensors[tensor_name];
            var offset = tensor.offset;

            byte* basePtr = null;
            mmf_accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            return (T*)(basePtr + offset);
        }

        public unsafe void load_tensor_data<T>(string tensor_name, Tensor<T> target, bool transpose = false)
            where T: unmanaged, INumber<T> {
            var tensor = tensors[tensor_name];

            if (transpose && target.rank != 2) {
                throw new Exception($"Only rank 2 tensors can be transposed!");
            }

            Console.WriteLine($"\nloading tensor {tensor_name} into target {target}");
            Console.WriteLine($"source: [{tensor.shape.join()}], linear_length: {tensor.linear_length:N0}");
            Console.WriteLine($"target: [{target.shape.join()}], linear_length: {target.linear_length:N0}");
            if (tensor.linear_length != target.linear_length) {
                throw new Exception(
                                    $"Tensor data size mismatch! {tensor.linear_length:N0} required, but target has {target.linear_length:N0})"
                                   );
            }

            if (!target.has_same_shape(transpose ? tensor.shape.Reverse() : tensor.shape)) {
                throw new Exception($"The tensor {tensor_name}:[{tensor.shape.join()}] does not have the required shape:[{target.shape.join()}]");
            }

            using var stream = mmf.CreateViewStream();
            stream.Seek((long)tensor.offset, SeekOrigin.Begin);


            if (transpose) {
                var data_ptr = load_tensor_data<T>(tensor_name);
                var source = TensorExtensions.calculate_strides(tensor.shape);

                for (var i = 0; i < tensor.shape[0]; ++i) {
                    for (var j = 0; j < tensor.shape[1]; ++j) {
                        target[[j, i]] = data_ptr[TensorExtensions.calculate_index(source.strides, i, j)];
                    }
                }
            } else {
                stream.ReadExactly(MemoryMarshal.AsBytes(target.as_span()));
            }
        }

        public void load_tensor_data<T>(string tensor_name, Span<T> target)
            where T: unmanaged {
            var tensor = tensors[tensor_name];

            using var stream = mmf.CreateViewStream();
            stream.Seek((long)tensor.offset, SeekOrigin.Begin);

            if (tensor.linear_length != (ulong)target.Length) {
                throw new InvalidDataException($"Tensor data size mismatch. Expected {tensor.linear_length:N0}, but got {target.Length:N0}");
            }

            stream.ReadExactly(MemoryMarshal.AsBytes(target));
        }

        public Tokenizer build_tokenizer() {
            var tokens = ((List<object>)metadata["tokenizer.ggml.tokens"]).Cast<string>().ToList();
            var ids    = Enumerable.Range(0, tokens.Count);

            // Use .Zip to combine tokens with their IDs
            var token_to_id = tokens.Zip(ids, static (token, id) => (token, id))
                                    .ToDictionary(static x => x.token, static x => x.id);

            var merges = ((List<object>)metadata["tokenizer.ggml.merges"]).Cast<string>().ToList();

            // Parse BPE merges into dictionary with ranks
            var bpe_ranks = new Dictionary<(string, string), int>();
            for (var i = 0; i < merges.Count; i++) {
                var parts = merges[i].Split(' ');
                bpe_ranks[(parts[0], parts[1])] = i;
            }

            return new Tokenizer(token_to_id, bpe_ranks);
        }

        public void Dispose() {
            mmf_accessor.Dispose();
            mmf.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    public sealed class TensorInfo {
        public required string name { get; init; }
        public required int[] shape { get; init; }
        public required int[] strides { get; init; }

        public required GGMLType type { get; init; }
        public required ulong offset { get; init; }
        public required ulong linear_length { get; init; }

        private ulong get_type_size(GGMLType type) {
            return type switch {
                GGMLType.F32 => 4,
                GGMLType.F16 => 2,
                GGMLType.Q4_0 or GGMLType.Q4_1 or GGMLType.Q5_0 or GGMLType.Q5_1 or GGMLType.Q8_0 => 1,
                GGMLType.Q8_1 => 2,
                GGMLType.Q2_K or GGMLType.Q3_K or GGMLType.Q4_K or GGMLType.Q5_K or GGMLType.Q6_K => 2,
                GGMLType.Q8_K => 8,
                GGMLType.I8 => 1,
                GGMLType.I16 => 2,
                GGMLType.I32 => 4,
                GGMLType.I64 => 8,
                GGMLType.F64 => 8,
                _ => throw new NotSupportedException($"Unsupported GGML type: {type}")
            };
        }

        public override string ToString()
            => $"{name} [{string.Join(',', shape)}], type:{type}, offset:{offset}, length:{linear_length}";
    }

    public enum GGMLType: uint {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        I8 = 24,
        I16 = 25,
        I32 = 26,
        I64 = 27,
        F64 = 28
    }

    enum MetadataValueType: uint {
        UINT8 = 0,
        INT8 = 1,
        UINT16 = 2,
        INT16 = 3,
        UINT32 = 4,
        INT32 = 5,
        FLOAT32 = 6,
        BOOL = 7,
        STRING = 8,
        ARRAY = 9,
        UINT64 = 10,
        INT64 = 11,
        FLOAT64 = 12
    }

    public class Tokenizer {
        private readonly Dictionary<string, int> token_to_id;
        private readonly Dictionary<int, string> id_to_token;
        private readonly Dictionary<(string, string), int> bpe_ranks;

        public Tokenizer(Dictionary<string, int> token_to_id, Dictionary<(string, string), int> bpe_ranks) {
            this.token_to_id = token_to_id;
            this.id_to_token = token_to_id.ToDictionary(static kvp => kvp.Value, static kvp => kvp.Key);
            this.bpe_ranks   = bpe_ranks;
        }

        public List<string> tokenize(string text) {
            var tokens = text.Split(' ').Select(token => bpe(token + "</w>")).ToList();
            return tokens.SelectMany(static token => token.Split(' ')).ToList();
        }

        public List<int> tokenize_and_convert_to_ids(string text) {
            var tokens = tokenize(text);
            return convert_tokens_to_ids(tokens);
        }

        private string bpe(string token) {
            var word  = token.ToCharArray().Select(static c => c.ToString()).ToList();
            var pairs = get_pairs(word);

            while (true) {
                if (pairs.Count == 0)
                    break;

                var bigram = pairs.OrderBy(pair => bpe_ranks.GetValueOrDefault(pair, int.MaxValue)).First();
                if (!bpe_ranks.ContainsKey(bigram))
                    break;

                var first   = bigram.Item1;
                var second  = bigram.Item2;
                var newWord = new List<string>();
                var i       = 0;

                while (i < word.Count) {
                    var j = word.IndexOf(first, i);
                    if (j == -1) {
                        newWord.AddRange(word.GetRange(i, word.Count - i));
                        break;
                    }

                    newWord.AddRange(word.GetRange(i, j - i));
                    i = j;

                    if (i < word.Count - 1 && word[i] == first && word[i + 1] == second) {
                        newWord.Add(first + second);
                        i += 2;
                    } else {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }

                word = newWord;
                if (word.Count == 1)
                    break;

                pairs = get_pairs(word);
            }

            return string.Join(" ", word);
        }

        private List<(string, string)> get_pairs(List<string> word) {
            var pairs = new List<(string, string)>();
            for (var i = 0; i < word.Count - 1; i++) {
                pairs.Add((word[i], word[i + 1]));
            }

            return pairs;
        }

        public List<int> convert_tokens_to_ids(List<string> tokens) {
            return tokens.Select(
                                 token => token_to_id.TryGetValue(token, out var value)
                                     ? value
                                     : token_to_id["<|endoftext|>"]
                                )
                         .ToList();
        }

        private string post_process_tokens(List<string> tokens) {
            var text = string.Join("", tokens);

            // Remove end of word tokens and add spaces
            text = text.Replace("</w>", " ");

            // Handle special tokens
            text = text.Replace("<|endoftext|>", "");

            // Trim any extra whitespace
            return text.Trim();
        }

        public List<int> encode(string text) {
            var tokens = tokenize(text);
            return convert_tokens_to_ids(tokens);
        }

        public string decode(List<int> ids) {
            var tokens = ids.Select(id => id_to_token.GetValueOrDefault(id, "<|unknown|>")).ToList();
            return post_process_tokens(tokens);
        }
    }
}