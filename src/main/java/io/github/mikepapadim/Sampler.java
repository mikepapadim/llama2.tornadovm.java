package io.github.mikepapadim;

/**
 * The Sampler class represents a simple sampler for various random sampling
 * techniques. It includes functionality for random integer and floating-point
 * number generation using the xorshift pseudo-random number generator. The
 * class is designed for use in sampling scenarios, particularly for natural
 * language processing tasks.
 */
public class Sampler {

    /**
     * The size of the vocabulary used in sampling.
     */
    final int vocab_size;

    /**
     * Buffer used in top-p sampling.
     */
    final int[] probindex;

    /**
     * The temperature parameter for controlling the randomness of the sampling
     * process. Higher values lead to more random samples.
     */
    final float temperature;

    /**
     * The top-p parameter for nucleus sampling, controlling the cumulative
     * probability mass to consider for sampling. Should be in the range (0, 1].
     */
    final float topp;

    /**
     * The seed for the pseudo-random number generator.
     */
    long rng_seed;

    /**
     * Constructs a Sampler with the specified parameters.
     *
     * @param vocab_size
     *            The size of the vocabulary used in sampling.
     * @param temperature
     *            The temperature parameter for controlling the randomness of the
     *            sampling process.
     * @param topp
     *            The top-p parameter for nucleus sampling, controlling the
     *            cumulative probability mass to consider for sampling. Should be in
     *            the range (0, 1].
     * @param rng_seed
     *            The seed for the pseudo-random number generator.
     */
    public Sampler(int vocab_size, float temperature, float topp, long rng_seed) {
        this.vocab_size = vocab_size;
        this.temperature = temperature;
        this.topp = topp;
        this.rng_seed = rng_seed;
        this.probindex = new int[vocab_size];
    }

    /**
     * Generates a random 32-bit integer using the xorshift pseudo-random number
     * generator.
     *
     * @return A random 32-bit integer.
     */
    int random_u32() {
        rng_seed ^= rng_seed >> 12;
        rng_seed ^= rng_seed << 25;
        rng_seed ^= rng_seed >> 27;
        return (int) ((rng_seed * 0x2545F4914F6CDD1DL) >> 32);
    }

    /**
     * Generates a random floating-point number in the range [0, 1) using the
     * xorshift pseudo-random number generator.
     *
     * @return A random floating-point number in the range [0, 1).
     */
    float random_f32() {
        return (random_u32() >>> 8) / 16777216.0f;
    }
}
