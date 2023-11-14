package io.github.mikepapadim;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;

public class Tokenizer {
    final String[] vocab;
    final float[] vocab_scores;
    final int vocab_size;
    final int max_token_length;
    Map<String, Integer> sorted_vocab;

    Tokenizer(String tokenizer_path, int vocab_size) throws IOException {
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
