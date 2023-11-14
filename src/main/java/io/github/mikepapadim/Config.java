package io.github.mikepapadim;

import java.nio.ByteBuffer;

public  class Config {
    final int dim; // transformer dimension
    final int hidden_dim; // for ffn layers
    final int n_layers; // number of layers
    final int n_heads; // number of query heads
    final int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    final int vocab_size; // vocabulary size, usually 256 (byte-level)
    final int seq_len; // max sequence length
    final boolean shared_weights;
    final int head_size;

    Config(ByteBuffer buffer) {
        this.dim = buffer.getInt();
        this.hidden_dim = buffer.getInt();
        this.n_layers = buffer.getInt();
        this.n_heads = buffer.getInt();
        this.n_kv_heads = buffer.getInt();
        int vocab_size = buffer.getInt();
        this.vocab_size = Math.abs(vocab_size);
        this.seq_len = buffer.getInt();
        this.shared_weights = vocab_size > 0;
        this.head_size = dim / n_heads;
    }

    @Override
    public String toString() {
        return "Config{" +
                "dim=" + dim +
                ", hidden_dim=" + hidden_dim +
                ", n_layers=" + n_layers +
                ", n_heads=" + n_heads +
                ", n_kv_heads=" + n_kv_heads +
                ", vocab_size=" + vocab_size +
                ", seq_len=" + seq_len +
                ", shared_weights=" + shared_weights +
                ", head_size=" + head_size +
                '}';
    }

}
