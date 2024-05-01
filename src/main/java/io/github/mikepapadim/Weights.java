package io.github.mikepapadim;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.tensors.DType;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.Tensor;
import uk.ac.manchester.tornado.api.types.tensors.TensorFP32;
//import uk.ac.manchester.tornado.api.types.t

/**
 * The Weights class represents the weight parameters of a Transformer model,
 * including various weight matrices for token embeddings, attention mechanisms,
 * feedforward networks, and classifier logits.
 */
public class Weights {

    // token embedding table
    final FloatBuffer token_embedding_table; // (vocab_size, dim)

    // weights for rmsnorms
    final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights

    // weights for matmuls. note dim == n_heads * head_size
    final TensorFP32[] wq; // (layer, dim, n_heads * head_size)
    final TensorFP32[] wk; // (layer, dim, n_kv_heads * head_size)
    final TensorFP32[] wv; // (layer, dim, n_kv_heads * head_size)
    final FloatBuffer[] wo; // (layer, n_heads * head_size, dim)

    // weights for ffn
    final FloatBuffer[] rms_ffn_weight; // (layer, dim)
    final FloatBuffer[] w1; // (layer, hidden_dim, dim)
    final FloatBuffer[] w2; // (layer, dim, hidden_dim)
    final FloatBuffer[] w3; // (layer, hidden_dim, dim)

    // final rmsnorm
    final FloatBuffer rms_final_weight; // (dim,)

    final FloatBuffer wcls; // (vocab_size, dim)

    TensorFP32 weightTensor; // vocabInTensor

    /**
     * Constructs Weights by parsing information from a checkpoint's memory segment.
     *
     * @param config
     *            The configuration of the Transformer model.
     * @param memorySegment
     *            The memory segment containing weight information.
     */
    Weights(Config config, MemorySegment memorySegment) {
        long[] position = new long[] { 0 };
        this.token_embedding_table = takeFloats(memorySegment, position, config.vocab_size, config.dim);
        this.rms_att_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.wq = takeTensors(memorySegment, position, config.n_layers, config.dim, config.n_heads * config.head_size);
        this.wk = takeTensors(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wv = takeTensors(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wo = takeArray(memorySegment, position, config.n_layers, config.n_heads * config.head_size, config.dim);
        this.rms_ffn_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.w1 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.w2 = takeArray(memorySegment, position, config.n_layers, config.dim, config.hidden_dim);
        this.w3 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.rms_final_weight = takeFloats(memorySegment, position, config.dim);
        position[0] += ((long) config.seq_len * config.head_size / 2) * Float.BYTES; // skip what used to be freq_cis_real (for RoPE)
        position[0] += ((long) config.seq_len * config.head_size / 2) * Float.BYTES; // skip what used to be freq_cis_imag (for RoPE)
        this.wcls = config.shared_weights ? this.token_embedding_table : takeFloats(memorySegment, position, config.vocab_size, config.dim);
        this.weightTensor = getWeightTensor(wcls, wcls.remaining());
    }

    /**
     * Creates and returns a {@code TensorFP32} object initialized with data from a given {@code FloatBuffer}.
     * The method constructs a new {@code Shape} object with the specified size to define the dimensions of the tensor.
     * It then copies the contents of the provided {@code FloatBuffer} into the tensor's memory segment.
     *
     * @param buffer the {@code FloatBuffer} containing the float data to be used in the tensor.
     * @param size   the size of the tensor, which determines the dimensions of the {@code Shape} used in tensor creation.
     * @return a new {@code TensorFP32} instance with data copied from the provided buffer and the specified size.
     */
    TensorFP32 getWeightTensor(FloatBuffer buffer, int size) {
        Shape shape = new Shape(size);
        TensorFP32 t = new TensorFP32(shape);
        t.getSegment().copyFrom(MemorySegment.ofBuffer(buffer));
        return t;
    }

    FloatBuffer takeFloats(MemorySegment memorySegment, long[] position, int... dims) {
        long totalBytes = 1;
        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        MemorySegment slice = memorySegment.asSlice(position[0], totalBytes);
        position[0] += totalBytes;
        return slice.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    FloatBuffer[] takeArray(MemorySegment memorySegment, long[] position, int dim0, int... dims) {
        FloatBuffer[] segments = new FloatBuffer[dim0];
        for (int i = 0; i < dim0; ++i) {
            segments[i] = takeFloats(memorySegment, position, dims);
        }
        return segments;
    }

    TensorFP32[] takeTensors(MemorySegment memorySegment, long[] position, int dim0, int... dims) {
        TensorFP32[] weightTensors = new TensorFP32[dim0];
        for (int i = 0; i < dim0; ++i) {
            weightTensors[i] = takeTensor(memorySegment, position, dims);
        }
        return weightTensors;
    }

    /**
     * Constructs a {@code TensorFP32} from a subsection of a {@code MemorySegment} based on specified dimensions.
     * This method calculates the total size in bytes required for the tensor based on the provided dimensions,
     * where each dimension's size multiplies the size of a float (since it operates in a float-specific context).
     * It then creates a slice from the original memory segment starting from a specified position and with the size
     * calculated. This slice is used to populate the new {@code TensorFP32} instance. After populating, the starting
     * position in the original memory segment is updated by the size of the slice.
     *
     * @param memorySegment the {@code MemorySegment} from which the tensor data will be extracted.
     * @param position      an array with the starting position for the slice in the memory segment. This value is updated
     *                      to reflect the new position after the slice is taken.
     * @param dims          variable number of integer arguments representing the dimensions of the tensor. These define
     *                      the shape and the amount of data to extract from the memory segment.
     * @return a new {@code TensorFP32} instance containing the data extracted from the specified segment of memory.
     */
    TensorFP32 takeTensor(MemorySegment memorySegment, long[] position, int... dims) {
        long totalBytes = 1;

        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        MemorySegment slice = memorySegment.asSlice(position[0], totalBytes);
        position[0] += totalBytes;
        long[] longArray = Arrays.stream(dims).mapToLong(i -> i).toArray();
        Shape shape = new Shape(longArray);
        TensorFP32 t = new TensorFP32(shape);
        t.getSegment().copyFrom(slice);
        return t;
    }

}