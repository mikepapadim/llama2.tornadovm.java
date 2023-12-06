package io.github.mikepapadim;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * The Tokenizer class represents a simple tokenizer that loads vocabulary and
 * scores from a file and provides methods for tokenization.
 */
public class Tokenizer {

    /**
     * The vocabulary array containing token strings.
     */
    final String[] vocab;

    /**
     * The array containing scores corresponding to each token in the vocabulary.
     */
    final float[] vocab_scores;

    /**
     * The size of the vocabulary.
     */
    final int vocab_size;

    /**
     * The maximum token length allowed by the tokenizer.
     */
    final int max_token_length;

    /**
     * A map containing the sorted vocabulary for efficient lookups.
     */
    Map<String, Integer> sorted_vocab;

    /**
     * Constructs a Tokenizer by loading vocabulary and scores from a file. Note: We
     * replaced the segment map from the original implementation as it caused a
     * Graal compiler issue in the TornadoVM.
     *
     * @param tokenizer_path
     *            The path to the tokenizer file.
     * @param vocab_size
     *            The size of the vocabulary.
     * @throws IOException
     *             If an I/O error occurs while reading the tokenizer file.
     */
    public Tokenizer(String tokenizer_path, int vocab_size) throws IOException {
        // i should have written the vocab_size into the tokenizer file... sigh
        this.vocab_size = vocab_size;

        // malloc space to hold the scores and the strings
        this.vocab = new String[vocab_size];
        this.vocab_scores = new float[vocab_size];

        // read in the file
        try (FileChannel channel = FileChannel.open(Paths.get(tokenizer_path), StandardOpenOption.READ)) {
            ByteBuffer tokBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            tokBuffer.order(ByteOrder.LITTLE_ENDIAN);
            this.max_token_length = tokBuffer.getInt();

            for (int i = 0; i < vocab_size; i++) {
                this.vocab_scores[i] = tokBuffer.getFloat();
                int len = tokBuffer.getInt();
                byte[] bytes = new byte[len];
                tokBuffer.get(bytes);
                this.vocab[i] = new String(bytes, StandardCharsets.UTF_8);
            }
        }
    }
}
