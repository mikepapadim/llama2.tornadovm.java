package io.github.mikepapadim;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat16;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat8;

/**
 * This class is used to maintain the state of the model during processing.
 */
public class RunState {
    // current wave of activations
    final float[] x; // activation at current time stamp (dim,)
    final float[] xb; // same, but inside a residual branch (dim,)
    final float[] xb2; // an additional buffer just for convenience (dim,)
    final float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    final float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    final float[] q; // query (dim,)
    final float[] k; // key (dim,)
    final float[] v; // value (dim,)
    final float[] att; // buffer for scores/attention values (n_heads, seq_len)
    final float[] logits; // output logits
    // kv cache
    final float[][] key_cache; // (layer, seq_len, dim)
    final float[][] value_cache; // (layer, seq_len, dim)

    final VectorFloat16 xVectorFloat16; // activation at current time stamp (dim,) in VectorFloat16
    final VectorFloat8 xVectorFloat8; // activation at current time stamp (dim,) in VectorFloat8
    final VectorFloat4 xVectorFloat4; // activation at current time stamp (dim,) in VectorFloat4
    final FloatArray xfa; // activation at current time stamp (dim,) in FloatArray

    /**
     * Constructs a {@code RunState} object using the provided {@link Config}.
     *
     * @param config The {@link Config} object containing transformer model configuration.
     */
    RunState(Config config) {
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        this.x = new float[config.dim];
        this.xVectorFloat16 = new VectorFloat16(config.dim / 16);
        this.xVectorFloat8 = new VectorFloat8(config.dim / 8);
        this.xVectorFloat4 = new VectorFloat4(config.dim / 4);
        this.xfa = new FloatArray(config.dim);
        this.xb = new float[config.dim];
        this.xb2 = new float[config.dim];
        this.hb = new float[config.hidden_dim];
        this.hb2 = new float[config.hidden_dim];
        this.q = new float[config.dim];
        this.k = new float[kv_dim];
        this.v = new float[kv_dim];
        this.att = new float[config.n_heads * config.seq_len];
        this.logits = new float[config.vocab_size];
        this.key_cache = new float[config.n_layers][config.seq_len * kv_dim];
        this.value_cache = new float[config.n_layers][config.seq_len * kv_dim];
    }
}