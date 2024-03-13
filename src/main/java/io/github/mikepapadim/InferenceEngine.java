package io.github.mikepapadim;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat16;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat8;
import uk.ac.manchester.tornado.api.types.vectors.Float16;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
import uk.ac.manchester.tornado.api.types.vectors.Float8;

/**
 * This class performs forward inference using a Transformer model.
 */
public class InferenceEngine {

    /**
     * Performs forward inference using the Transformer model.
     *
     * @param transformer
     *            The Transformer model to use for inference.
     * @param token
     *            The input token for processing.
     * @param pos
     *            The position of the input token in the sequence.
     * @param executionPlan
     *            The list of TornadoExecutionPlan objects for execution.
     * @return The output logits produced by the Transformer model.
     */
    static float[] forward(Transformer transformer, int token, int pos, TornadoExecutionPlan executionPlan) {
        Config p = transformer.config;
        Weights w = transformer.weights;
        RunState s = transformer.state;
        int dim = p.dim;
        int hidden_dim = p.hidden_dim;
        int head_size = p.head_size;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        // copy the token embedding into x
        w.token_embedding_table.get(token * dim, s.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {

            // attention rmsnorm
            rmsnorm(s.xb, s.x, w.rms_att_weight[l], dim);

            // qkv matmuls for this position
            MatrixVectorCollection.matmul(s.q, s.xb, w.wq[l], dim, dim);
            MatrixVectorCollection.matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
            MatrixVectorCollection.matmul(s.v, s.xb, w.wv[l], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0 / Math.pow(10000.0f, head_dim / (float) head_size));
                float val = pos * freq;
                float fcr = (float) Math.cos(val);
                float fci = (float) Math.sin(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    float[] vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // save key,value at this time step (pos) to our kv cache
            // int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            System.arraycopy(s.k, 0, s.key_cache[l], pos * kv_dim, kv_dim);
            System.arraycopy(s.v, 0, s.value_cache[l], pos * kv_dim, kv_dim);

            final int curLayer = l;

            // multihead attention. iterate over all heads
            IntStream.range(0, p.n_heads).parallel().forEach(h -> {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                int qOffset = h * head_size;

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                int attOffset = h * p.seq_len;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyCacheOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[qOffset + i] * s.key_cache[curLayer][keyCacheOffset + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                int xbOffset = h * head_size;
                // memset(xb, 0, head_size * sizeof(float));
                Arrays.fill(s.xb, xbOffset, xbOffset + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int vOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attOffset + t];
                    // accumulate the weighted value inconfigto xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbOffset + i] += a * s.value_cache[curLayer][vOffset + i];
                    }
                }
            });

            // final matmul to get the output of the attention
            MatrixVectorCollection.matmul(s.xb2, s.xb, w.wo[l], dim, dim);

            residualConnection(s.x, s.xb2, dim);

            // ffn rmsnorm
            rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            MatrixVectorCollection.matmul(s.hb, s.xb, w.w1[l], dim, p.hidden_dim);
            MatrixVectorCollection.matmul(s.hb2, s.xb, w.w3[l], dim, p.hidden_dim);

            fusedSiluEwiseMul(hidden_dim, s.hb, s.hb2);

            // final matmul to get the output of the ffn
            MatrixVectorCollection.matmul(s.xb, s.hb, w.w2[l], p.hidden_dim, dim);

            residualConnection(s.x, s.xb, dim);
        }

        // final rmsnorm
        rmsnorm(s.x, s.x, w.rms_final_weight, dim);
        // MatrixVectorCollection.matmul(s.logits, s.x, w.wcls, dim, p.vocab_size);
        // MatrixVectorCollection.matmul(s.logits, s.x, w.weightTensor, dim,
        // p.vocab_size);

        executionPlan.execute();

        return s.logits;
    }

    /**
     * Converts a primitive float array to a TornadoVM {@link VectorFloat16}.
     * 
     * @param destination
     *            The {@link VectorFloat16} that will store the data of the float
     *            array.
     * @param source
     *            The input data for the {@link VectorFloat16}.
     */
    static void convertToVectorFloat16(VectorFloat16 destination, float[] source) {
        int numVectors = source.length / destination.vectorWidth();
        for (int i = 0; i < numVectors; i++) {
            Float16 float16 = new Float16();
            for (int j = 0; j < destination.vectorWidth(); j++) {
                float16.set(j, source[i * destination.vectorWidth() + j]);
            }
            destination.set(i, float16);
        }
    }

    /**
     * Converts a primitive float array to a TornadoVM {@link VectorFloat8}.
     * 
     * @param destination
     *            The {@link VectorFloat8} that will store the data of the float
     *            array.
     * @param source
     *            The input data for the {@link VectorFloat8}.
     */
    static void convertToVectorFloat8(VectorFloat8 destination, float[] source) {
        int numVectors = source.length / 8;
        for (int i = 0; i < numVectors; i++) {
            Float8 float8 = new Float8();
            for (int j = 0; j < 8; j++) {
                float8.set(j, source[i * 8 + j]);
            }
            destination.set(i, float8);
        }
    }

    /**
     * Converts a primitive float array to a TornadoVM {@link VectorFloat4}.
     * 
     * @param destination
     *            The {@link VectorFloat4} that will store the data of the float
     *            array.
     * @param source
     *            The input data for the {@link VectorFloat4}.
     */
    static void convertToVectorFloat4(VectorFloat4 destination, float[] source) {
        int numVectors = source.length / 4;
        for (int i = 0; i < numVectors; i++) {
            Float4 float4 = new Float4();
            for (int j = 0; j < 4; j++) {
                float4.set(j, source[i * 4 + j]);
            }
            destination.set(i, float4);
        }
    }

    /**
     * Applies the SwiGLU non-linearity to the output of the first layer in the
     * Transformer model.
     *
     * @param hidden_dim
     *            The hidden dimension of the model.
     * @param out
     *            The output array to store the result.
     * @param hb2
     *            The array representing the hidden layer output.
     */
    static void fusedSiluEwiseMul(int hidden_dim, float[] out, float[] hb2) {
        for (int i = 0; i < hidden_dim; i++) {
            float val = out[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + Math.exp(-val)));
            // elementwise multiply with w3(x)
            out[i] = val * hb2[i];
        }
    }

    /**
     * Applies the residual connection by element-wise addition of the input and
     * residual vectors.
     *
     * @param s
     *            The input vector.
     * @param xb2
     *            The residual vector to be added.
     * @param dim
     *            The dimension of the vectors.
     */
    static void residualConnection(float[] s, float[] xb2, int dim) {
        for (int i = 0; i < dim; i++) {
            s[i] = s[i] + xb2[i];
        }
    }

    /**
     * Applies root mean square normalization to the input vector.
     *
     * @param o
     *            The output vector.
     * @param x
     *            The input vector.
     * @param weight
     *            The weight values for normalization.
     * @param size
     *            The size of the vectors.
     */
    static void rmsnorm(float[] o, float[] x, FloatBuffer weight, int size) {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight.get(j) * (ss * x[j]);
        }
    }

    /**
     * Applies the softmax function to a portion of the input array.
     *
     * @param x
     *            The input array.
     * @param xOffset
     *            The offset within the input array.
     * @param size
     *            The size of the portion to apply softmax.
     */
    static void softmax(float[] x, int xOffset, int size) {
        // find max value (for numerical stability)
        float max_val = x[0 + xOffset];
        for (int i = 1; i < size; i++) {
            if (x[i + xOffset] > max_val) {
                max_val = x[i + xOffset];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i + xOffset] = (float) Math.exp(x[i + xOffset] - max_val);
            sum += x[i + xOffset];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x[i + xOffset] /= sum;
        }
    }
}
