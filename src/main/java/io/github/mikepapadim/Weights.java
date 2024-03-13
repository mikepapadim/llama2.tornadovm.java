package io.github.mikepapadim;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.tensors.Tensor;
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

    FloatArray weightInFloatArray; // vocab in FloatArray

    Tensor weightTensor; // vocabInTensor

    // Tensor
    // Tensor
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

        this.weightTensor = Tensor.fromArray(wclsAsPrimitive);
        //
        // this.weightInFloatArray = FloatArray.fromArray(wclsAsPrimitive);

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

}