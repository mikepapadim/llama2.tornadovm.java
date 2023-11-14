package io.github.mikepapadim;

public class  Sampler {
    final int vocab_size;
    final int[] probindex; // buffer used in top-p sampling
    final float temperature;
    final float topp;
    long rng_seed;

    Sampler(int vocab_size, float temperature, float topp, long rng_seed) {
        this.vocab_size = vocab_size;
        this.temperature = temperature;
        this.topp = topp;
        this.rng_seed = rng_seed;
        // buffer only used with nucleus sampling; may not need but it's ~small
        this.probindex = new int[vocab_size];
    }

    int random_u32() {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        rng_seed ^= rng_seed >> 12;
        rng_seed ^= rng_seed << 25;
        rng_seed ^= rng_seed >> 27;
        return (int) ((rng_seed * 0x2545F4914F6CDD1DL) >> 32);
    }

    float random_f32() { // random float32 in [0,1)
        return (random_u32() >>> 8) / 16777216.0f;
    }
}