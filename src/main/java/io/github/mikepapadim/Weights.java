package io.github.mikepapadim;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat16;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat8;
import uk.ac.manchester.tornado.api.types.vectors.Float16;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
import uk.ac.manchester.tornado.api.types.vectors.Float8;

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
    final FloatBuffer[] wq; // (layer, dim, n_heads * head_size)
    final FloatBuffer[] wk; // (layer, dim, n_kv_heads * head_size)
    final FloatBuffer[] wv; // (layer, dim, n_kv_heads * head_size)
    final FloatBuffer[] wo; // (layer, n_heads * head_size, dim)

    // weights for ffn
    final FloatBuffer[] rms_ffn_weight; // (layer, dim)
    final FloatBuffer[] w1; // (layer, hidden_dim, dim)
    final FloatBuffer[] w2; // (layer, dim, hidden_dim)
    final FloatBuffer[] w3; // (layer, hidden_dim, dim)

    // final rmsnorm
    final FloatBuffer rms_final_weight; // (dim,)

    // (optional) classifier weights for the logits, on the last layer
    final FloatBuffer wcls; // (vocab_size, dim)
    float[] wclsAsPrimitive;

    // Data structures for TornadoVM
    VectorFloat16 weightInVectorFloat16; // vocab in VectorFloat16
    VectorFloat8 weightInVectorFloat8; // vocab in VectorFloat8
    VectorFloat4 weightInVectorFloat4; // vocab in VectorFloat4
    FloatArray weightInFloatArray; // vocab in FloatArray

    ArrayList<float[]> weightsAsPrimitivesK;
    ArrayList<float[]> weightsAsPrimitivesV;
    ArrayList<float[]> weightsAsPrimitivesQ;

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
        this.wq = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_heads * config.head_size);
        this.wk = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wv = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wo = takeArray(memorySegment, position, config.n_layers, config.n_heads * config.head_size, config.dim);
        this.rms_ffn_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.w1 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.w2 = takeArray(memorySegment, position, config.n_layers, config.dim, config.hidden_dim);
        this.w3 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.rms_final_weight = takeFloats(memorySegment, position, config.dim);
        position[0] += (config.seq_len * config.head_size / 2) * Float.BYTES; // skip what used to be freq_cis_real (for RoPE)
        position[0] += (config.seq_len * config.head_size / 2) * Float.BYTES; // skip what used to be freq_cis_imag (for RoPE)
        this.wcls = config.shared_weights ? this.token_embedding_table : takeFloats(memorySegment, position, config.vocab_size, config.dim);

        // Convert FloatBuffer to primitive float
        this.wclsAsPrimitive = new float[wcls.remaining()];
        wcls.get(wclsAsPrimitive);

        // Convert the read-only weights used in the last mat-mul to TornadoVM datatypes
        // that use MemorySegments
        this.weightInFloatArray = FloatArray.fromArray(wclsAsPrimitive);
        this.weightInVectorFloat16 = createVectorFloat16Array(weightInFloatArray);
        this.weightInVectorFloat8 = createVectorFloat8Array(weightInFloatArray);
        this.weightInVectorFloat4 = createVectorFloat4Array(weightInFloatArray);

        this.weightsAsPrimitivesK = normalizeInputWeight(wk);
        this.weightsAsPrimitivesV = normalizeInputWeight(wv);
        this.weightsAsPrimitivesQ = normalizeInputWeight(wq);
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

    ArrayList<float[]> normalizeInputWeight(FloatBuffer[] x) {
        ArrayList<float[]> xn = new ArrayList<>();

        for (FloatBuffer floatBuffer : x) {
            FloatBuffer src = floatBuffer.duplicate();
            float[] temp = new float[src.remaining()];
            src.get(temp);
            xn.add(temp);
        }
        return xn;
    }

    /**
     * Creates a TornadoVM VectorFloat16 array from a given FloatArray.
     *
     * @param fa
     *            The FloatArray to convert into VectorFloat16.
     * @return VectorFloat16 array containing the converted data.
     */
    private VectorFloat16 createVectorFloat16Array(FloatArray fa) {
        int numElements = fa.getSize();
        int numFloat16Vectors = numElements / 16;

        // Create an array to store the VectorFloat16 vectors
        VectorFloat16 vectorFloat16Array = new VectorFloat16(numFloat16Vectors);

        // Iterate over fa to create VectorFloat16 vectors
        for (int i = 0; i < numFloat16Vectors; i++) {
            // Extract a subset of sixteen elements from fa
            Float16 float16 = new Float16();
            for (int j = 0; j < 16; j++) {
                float16.set(j, fa.get(i * 16 + j));
            }

            // Create a VectorFloat16 using the extracted Float16
            vectorFloat16Array.set(i, float16);
        }

        return vectorFloat16Array;
    }

    /**
     * Creates a TornadoVM VectorFloat8 array from a given FloatArray.
     *
     * @param fa
     *            The FloatArray to convert into VectorFloat8.
     * @return VectorFloat8 array containing the converted data.
     */
    private VectorFloat8 createVectorFloat8Array(FloatArray fa) {
        int numElements = fa.getSize();
        int numFloat8Vectors = numElements / 8;

        // Create an array to store the VectorFloat8 vectors
        VectorFloat8 vectorFloat8Array = new VectorFloat8(numFloat8Vectors);

        // Iterate over fa to create VectorFloat8 vectors
        for (int i = 0; i < numFloat8Vectors; i++) {
            // Extract a subset of eight elements from fa
            Float8 float8 = new Float8();
            for (int j = 0; j < 8; j++) {
                float8.set(j, fa.get(i * 8 + j));
            }

            // Create a VectorFloat8 using the extracted Float8
            vectorFloat8Array.set(i, float8);
        }

        return vectorFloat8Array;
    }

    /**
     * Creates a TornadoVM VectorFloat4 array from a given FloatArray.
     *
     * @param fa
     *            The FloatArray to convert into VectorFloat4.
     * @return VectorFloat4 array containing the converted data.
     */
    private VectorFloat4 createVectorFloat4Array(FloatArray fa) {
        int numElements = fa.getSize();
        int numFloat4Vectors = numElements / 4;

        // Create an array to store the VectorFloat4 vectors
        VectorFloat4 vectorFloat4Array = new VectorFloat4(numFloat4Vectors);

        // Iterate over fa to create VectorFloat4 vectors
        for (int i = 0; i < numFloat4Vectors; i++) {
            // Extract a subset of four elements from fa
            Float4 float4 = new Float4();
            for (int j = 0; j < 4; j++) {
                float4.set(j, fa.get(i * 4 + j));
            }

            // Create a VectorFloat4 using the extracted Float4
            vectorFloat4Array.set(i, float4);
        }

        return vectorFloat4Array;
    }

}