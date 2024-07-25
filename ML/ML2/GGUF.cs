using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;

namespace NetML.ML2;

public class GGUFFile: IDisposable {
    public uint version { get; }
    public ulong tensor_count { get; }
    public ulong metadata_count { get; }
    public Dictionary<string, object> metadata { get; }
    public Dictionary<string, TensorInfo> tensors { get; }
    public uint alignment { get; }

    private const uint GGUF_MAGIC = 0x46554747;
    private readonly MemoryMappedFile mmf;
    private readonly MemoryMappedViewAccessor mmf_accessor;

    private GGUFFile(MemoryMappedFile mmf) {
        this.mmf = mmf;
        using var reader = new BinaryReader(mmf.CreateViewStream());

        var magic   = reader.ReadUInt32();
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

    public static GGUFFile from_file(string filePath) {
        var mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open);

        return new GGUFFile(mmf);
    }

    private static Dictionary<string, object> read_metadata(BinaryReader reader, ulong count) {
        var metadata = new Dictionary<string, object>();
        for (ulong i = 0; i < count; i++) {
            var key = read_string(reader);
            var value = read_value(reader);
            metadata[key] = value;
        }
        return metadata;
    }

    private static Dictionary<string, TensorInfo> read_tensor_info(BinaryReader reader, ulong count, uint alignment) {
        var tensors = new Dictionary<string, TensorInfo>();
        for (ulong i = 0; i < count; i++) {
            var tensor = new TensorInfo {
                                            name = read_string(reader),
                                            dimensions = read_uint64_array(reader),
                                            type = (GGMLType)reader.ReadUInt32(),
                                            offset = reader.ReadUInt64()
                                        };
            tensors.Add(tensor.name, tensor);
        }

        // Align to the specified alignment
        var currentPosition = reader.BaseStream.Position;
        var alignedPosition = ((currentPosition + alignment - 1) / alignment) * alignment;
        reader.BaseStream.Seek(alignedPosition, SeekOrigin.Begin);

        return tensors;
    }

    private static string read_string(BinaryReader reader) {
        var length = reader.ReadUInt64();
        var stringBytes = reader.ReadBytes((int)length);
        return Encoding.UTF8.GetString(stringBytes);
    }

    private static object read_value(BinaryReader reader, GGUFMetadataValueType? type = null) {
        if (type is null) {
            type = (GGUFMetadataValueType)reader.ReadUInt32();
        }

        return type switch {
            GGUFMetadataValueType.UINT8   => reader.ReadByte(),
            GGUFMetadataValueType.INT8    => reader.ReadSByte(),
            GGUFMetadataValueType.UINT16  => reader.ReadUInt16(),
            GGUFMetadataValueType.INT16   => reader.ReadInt16(),
            GGUFMetadataValueType.UINT32  => reader.ReadUInt32(),
            GGUFMetadataValueType.INT32   => reader.ReadInt32(),
            GGUFMetadataValueType.FLOAT32 => reader.ReadSingle(),
            GGUFMetadataValueType.BOOL    => reader.ReadBoolean(),
            GGUFMetadataValueType.STRING  => read_string(reader),
            GGUFMetadataValueType.ARRAY   => read_array(reader),
            GGUFMetadataValueType.UINT64  => reader.ReadUInt64(),
            GGUFMetadataValueType.INT64   => reader.ReadInt64(),
            GGUFMetadataValueType.FLOAT64 => reader.ReadDouble(),
            _                             => throw new NotSupportedException($"Unsupported metadata value type: {type}")
        };
    }

    private static List<object> read_array(BinaryReader reader) {
        var elementType = (GGUFMetadataValueType)reader.ReadUInt32();
        var count = reader.ReadUInt64();
        var array = new List<object>();
        for (ulong i = 0; i < count; i++) {
            array.Add(read_value(reader, elementType));
        }
        return array;
    }

    private static int[] read_uint64_array(BinaryReader reader) {
        var count = reader.ReadUInt32();
        return Enumerable.Range(0, (int)count).Select(_ => (int)reader.ReadUInt64()).ToArray();
    }

    private T get_meta_data_value<T>(string key, T defaultValue = default) {
        if (metadata.TryGetValue(key, out var value)) {
            if (value is T tValue) {
                return tValue;
            }
            return (T)Convert.ChangeType(value, typeof(T));
        }
        return defaultValue;
    }

    public unsafe NetML.ML2.Tensor<float> load_tensor(string tensor_name) {
        var tensor = tensors[tensor_name];

        return NetML.ML2.Tensor<float>.create(
                                              tensor.name,
                                              load_tensor_data<float>(tensor.name),
                                              (int)tensor.linear_length,
                                              tensor.dimensions
                                             );
    }

    public unsafe T* load_tensor_data<T>(string tensor_name) where T : unmanaged {
        var tensor = tensors[tensor_name];
        var offset    = tensor.offset;

        byte* basePtr = null;
        mmf_accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
        return (T*)(basePtr + offset);;
    }

    public void load_tensor_data<T>(string tensor_name, Span<T> target) where T: unmanaged {
        var tensor = tensors[tensor_name];

        using var stream = mmf.CreateViewStream();
        stream.Seek((long)tensor.offset, SeekOrigin.Begin);

        if (tensor.linear_length != (ulong)target.Length) {
            throw new InvalidDataException($"Tensor data size mismatch.");
        }

        stream.ReadExactly(MemoryMarshal.AsBytes(target));
    }

    public void Dispose() {
        mmf_accessor.Dispose();
        mmf.Dispose();
        GC.SuppressFinalize(this);
    }
}

public sealed class TensorInfo {
    public required string name { get; init; }
    public required int[] dimensions { get; init; }

    public required GGMLType type { get; init; }
    public required ulong offset { get; init; }
    public ulong linear_length => (ulong)(dimensions.Aggregate(1, static (a, b) => a * b) * (int)get_type_size(type));

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
        => $"{name} [{string.Join(',', dimensions)}], type:{type}, offset:{offset}, length:{linear_length}";
}

public enum GGMLType : uint {
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

enum GGUFMetadataValueType : uint {
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