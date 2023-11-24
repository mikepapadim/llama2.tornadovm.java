package io.github.mikepapadim;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat8;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
import uk.ac.manchester.tornado.api.types.vectors.Float8;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;

public class Weights {
    // token embedding table
    final FloatBuffer token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls. note dim == n_heads * head_size
    final FloatBuffer[] wq; // (layer, dim, n_heads * head_size)

    ArrayList<float[]> weights_rc;
    final FloatBuffer[] wk; // (layer, dim, n_kv_heads * head_size)
    final FloatBuffer[] wv; // (layer, dim, n_kv_heads * head_size)
    final FloatBuffer[] wo; // (layer, n_heads * head_size, dim)
    final FloatBuffer[] rms_ffn_weight; // (layer, dim)
    // weights for ffn
    final FloatBuffer[] w1; // (layer, hidden_dim, dim)
    final FloatBuffer[] w2; // (layer, dim, hidden_dim)
    final FloatBuffer[] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    final FloatBuffer rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    final FloatBuffer wcls; // (vocab_size, dim)

     float[] wclsAsPrimitive;

     VectorFloat8 wclsAsPrimitiveV;

     VectorFloat8 vectorFloat8Array;
     FloatArray fa;

    ArrayList<float[]> weightsAsPrimitivesK;
    ArrayList<float[]> weightsAsPrimitivesV;
    ArrayList<float[]> weightsAsPrimitivesQ;

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

    FloatBuffer[] takeArray2(MemorySegment memorySegment, long[] position, int dim0, int... dims) {
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

    // ----------------------------------------------------------------------------
    // initialization: read from checkpoint

    Weights(Config config, MemorySegment memorySegment) {
        long[] position = new long[]{0};
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
        this.wcls = config.shared_weights
                ? this.token_embedding_table
                : takeFloats(memorySegment, position, config.vocab_size, config.dim);
        this.wclsAsPrimitive = new float[wcls.remaining()];
        wcls.get(wclsAsPrimitive);

       this.fa = FloatArray.fromArray(wclsAsPrimitive);
      final int sizee = fa.getSize()/8;
        this.wclsAsPrimitiveV = new VectorFloat8(9216000/8);


        System.out.println("fa " + fa.getSize());
        System.out.println("vECTOR " + wclsAsPrimitiveV.getLength());

        int faSize = fa.getSize();

//        for (int i = 0; i < faSize/8; i += 8) {
////            System.out.println("i " + i);
//
//            // Ensure that you don't go out of bounds
////            if (i + 7 < faSize) {
////            System.out.println("i " + i);
//                wclsAsPrimitiveV.set(i , new Float8(fa.get(i), fa.get(i + 1), //
//                                             fa.get(i + 2), fa.get(i + 3), //
//                                             fa.get(i + 4), fa.get(i + 5), //
//                                             fa.get(i + 6), fa.get(i + 7)));
////            }
//        }

        int numElements = fa.getSize();
        int numFloat8Vectors = numElements / 8;

        // Create an array to store the Float8 vectors
         this.vectorFloat8Array = new VectorFloat8(numFloat8Vectors);

        // Iterate over fa to create Float8 vectors
        for (int i = 0; i < numFloat8Vectors; i++) {
            // Extract a subset of eight elements from fa
            Float8 float8 = new Float8();
            for (int j = 0; j < 8; j++) {
                float8.set(j, fa.get(i * 8 + j));
            }

            // Create a VectorFloat8 using the extracted Float8
            vectorFloat8Array.set(i,float8);
        }

        this.weightsAsPrimitivesK = normalizeInputWeight(wk);
        this.weightsAsPrimitivesV = normalizeInputWeight(wv);
        this.weightsAsPrimitivesQ = normalizeInputWeight(wq);

    }
}