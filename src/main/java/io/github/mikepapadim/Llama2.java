package io.github.mikepapadim;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * This class provides an implementation of the Llama-2 neural network model in
 * {@link https://github.com/mukel/llama2.java}, extended to offload the
 * computational heavy parts on heterogeneous devices using TornadoVM. It
 * includes methods for generating text, tokenization, and sampling.
 * Additionally, it offers four different execution modes, one using the Vector
 * API, and three TornadoVM modes, with Vector4, Vector8 and Vector16 types.
 */
class Llama2 {

    /**
     * Static variables that represent the different execution modes. By default the
     * Vector API is enabled, but this can be configured through the respective
     * system properties.
     */
    static final boolean USE_VECTOR_API = getBooleanProperty("VectorAPI", true);
    static final boolean USE_VECTORFLOAT4 = getBooleanProperty("VectorFloat4", false);
    static final boolean USE_VECTORFLOAT8 = getBooleanProperty("VectorFloat8", false);
    static final boolean USE_VECTORFLOAT16 = getBooleanProperty("VectorFloat16", false);

    private static boolean getBooleanProperty(String propertyName, boolean defaultValue) {
        return "true".equalsIgnoreCase(System.getProperty("llama2." + propertyName, String.valueOf(defaultValue)));
    }

    // ============= Utility functions =============

    /**
     * Creates the appropriate TornadoVM execution plan based on the selected
     * execution mode (i.e., VectorFloat4, VectorFloat8, VectorFloat16).
     *
     * @param transformer
     *            The Transformer model used for sequence generation.
     * @return The TornadoVM execution plan for the specified execution mode
     */
    private static TornadoExecutionPlan createTornadoExecutionPlan(Transformer transformer) {
        Config p = transformer.config;
        Weights w = transformer.weights;
        RunState s = transformer.state;
        int dim = p.dim;
        TaskGraph taskGraph;
        if (USE_VECTORFLOAT8) {
            taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, s.xVectorFloat8) //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, w.weightInVectorFloat8)//
                    .task("t0", MatrixVectorCollection::matrixVectorFloat8, s.logits, s.xVectorFloat8, w.weightInVectorFloat8, dim, transformer.config.vocab_size) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, s.logits);
        } else if (USE_VECTORFLOAT16) {
            taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, s.xVectorFloat16) //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, w.weightInVectorFloat16) //
                    .task("t0", MatrixVectorCollection::matrixVectorFloat16, s.logits, s.xVectorFloat16, w.weightInVectorFloat16, dim, transformer.config.vocab_size) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, s.logits);
        } else if (USE_VECTORFLOAT4) {
            taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, s.xVectorFloat4) //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, w.weightInVectorFloat4) //
                    .task("t0", MatrixVectorCollection::matrixVectorFloat4, s.logits, s.xVectorFloat4, w.weightInVectorFloat4, dim, transformer.config.vocab_size) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, s.logits);
        } else {
            taskGraph = new TaskGraph("s0") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, s.x) //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, w.weightInFloatArray) //
                    .task("t0", MatrixVectorCollection::matrixVectorSimple, s.logits, s.x, w.weightInFloatArray, dim, transformer.config.vocab_size) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, s.logits);
        }

        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    /**
     * Function used for benchmarking that returns the model speed.
     *
     * @return Returns the model speed in milliseconds.
     */
    static long time_in_ms() {
        return System.nanoTime() / 1_000_000;
    }

    /**
     * Used for printing the printable characters and whitespaces.
     *
     * @param piece
     *            The tokens to be printed.
     */
    static void safe_printf(String piece) {
        if (piece == null || piece.isEmpty()) {
            return;
        }

        if (piece.length() == 1) {
            char ch = piece.charAt(0);
            boolean isPrintable = (32 <= ch && ch < 127);
            if (!(isPrintable || Character.isWhitespace(ch))) {
                return;
            }
        }

        System.out.print(piece);
    }

    // ============= Computation =============

    /**
     * Byte Pair Encoding (BPE) Tokenizer that translates strings to tokens,
     * considering the previous token in the sequence. It performs additional
     * processing to handle specific cases, such as stripping leading whitespace
     * after a Beginning of Sentence (BOS) token or decoding raw byte tokens.
     *
     * @param t
     *            The Tokenizer containing the vocabulary and necessary decoding
     *            information.
     * @param prev_token
     *            The token preceding the current token in the sequence.
     * @param token
     *            The token to decode.
     * @return The decoded string representation of the token, considering the
     *         context of the previous token and handling special cases like raw
     *         byte tokens.
     */
    static String decode(Tokenizer t, int prev_token, int token) {
        String piece = t.vocab[token];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace
        if (prev_token == 1 && piece.charAt(0) == ' ') {
            piece = piece.substring(1);
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        String prefix = "<0x";
        String suffix = ">";
        if (piece.length() == 6 && piece.startsWith(prefix) && piece.endsWith(suffix)) {
            String hex2 = piece.substring(prefix.length(), prefix.length() + 2);
            char ch = (char) Integer.parseInt(hex2, 16);
            // only print printable chars or whitespace, excluding control codes, backspace,
            // etc.
            piece = Character.toString(ch);
        }
        return piece;
    }

    /**
     * Efficiently finds the perfect match for a given string in a sorted
     * vocabulary.
     *
     * @param str
     *            The string to look up in the vocabulary.
     * @param sorted_vocab
     *            A map representing the sorted vocabulary, where keys are strings
     *            and values are corresponding indices.
     * @return The index of the exact match in the vocabulary, or -1 if not found.
     */
    static int str_lookup(String str, Map<String, Integer> sorted_vocab) {
        return sorted_vocab.getOrDefault(str, -1);
    }

    /**
     * Encodes a given text into a sequence of tokens using the provided Tokenizer,
     * with optional Beginning of Sentence (BOS) and End of Sentence (EOS) tokens.
     *
     * @param t
     *            The Tokenizer containing the vocabulary and scoring information.
     * @param text
     *            The text to be encoded into tokens.
     * @param bos
     *            A boolean indicating whether to prepend a BOS token (true) or not
     *            (false).
     * @param eos
     *            A boolean indicating whether to append an EOS token (true) or not
     *            (false).
     * @param tokens
     *            An array preallocated to store the resulting sequence of tokens.
     * @return The number of tokens in the encoded sequence.
     *
     * @throws RuntimeException
     *             if the input text is null.
     */
    static int encode(Tokenizer t, String text, boolean bos, boolean eos, int[] tokens) {

        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS
        // token (=2)
        if (text == null) {
            System.err.println("cannot encode NULL text");
            System.exit(1);
        }

        if (t.sorted_vocab == null) {
            // sort vocabulary
            t.sorted_vocab = new HashMap<>();
            for (int i = 0; i < t.vocab_size; i++) {
                assert !t.sorted_vocab.containsKey(t.vocab[i]);
                t.sorted_vocab.put(t.vocab[i], i);
            }
        }

        // start at 0 tokens
        int n_tokens = 0; // the number of tokens

        // add optional BOS (=1) token, if desired
        if (bos) {
            tokens[n_tokens++] = 1;
        }

        // prepend a dummy prefix token to the input string, but only if text != ""
        if (!"".equals(text)) {
            int dummy_prefix = str_lookup(" ", t.sorted_vocab);
            tokens[n_tokens++] = dummy_prefix;
        }

        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = str_lookup(singleCodepoint, t.sorted_vocab);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens[n_tokens++] = id;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens[n_tokens++] = Byte.toUnsignedInt(b) + 3;
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in
        // vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
                int id = str_lookup(str_buffer, t.sorted_vocab);
                if (id != -1 && t.vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = t.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--; // token length decreased
        }

        // add optional EOS (=2) token, if desired
        if (eos) {
            tokens[n_tokens++] = 2;
        }

        return n_tokens;
    }

    // Sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

    /**
     * Samples an index from an array of probabilities.
     *
     * @param probabilities
     *            An array of probabilities. The sum of all probabilities must be 1.
     * @param n
     *            The number of elements in the probabilities array.
     * @param coin
     *            A random value between 0 and 1 used for sampling.
     * @return The index sampled based on the provided probabilities.
     */
    static int sample_mult(float[] probabilities, int n, float coin) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    /**
     * Swaps two elements from a specified array.
     *
     * @param array
     *            The array for which the elements will be swapped.
     * @param from
     *            The index of the first element to be swapped.
     * @param to
     *            The index of the second element to be swapped.
     */
    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    /**
     * Performs a sift-down on a specified integer array.
     *
     * @param array
     *            The integer array on which the sift-down will be performed.
     * @param from
     *            The index from which to start the sift-down operation.
     * @param n
     *            The number of elements.
     * @param comparator
     *            The comparator to determine the order of elements.
     */
    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from,next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    /**
     * Performs top-p sampling (or "nucleus sampling") to sample from a set of
     * tokens that exceed a certain probability threshold (topp). This method
     * ensures that tokens with very low probabilities are less likely to be
     * sampled.
     *
     * @param probabilities
     *            The array of probabilities for each token.
     * @param n
     *            The number of tokens.
     * @param topp
     *            The probability threshold for sampling.
     * @param indices
     *            An array to store the indices of the sampled tokens.
     * @param coin
     *            A random number in [0, 1), usually obtained from random_f32().
     * @return The index of the sampled token.
     */
    static int sample_topp(float[] probabilities, int n, float topp, int[] indices, float coin) {
        Comparator<Integer> comparator = Comparator.<Integer>comparingDouble(i -> probabilities[i]).reversed();

        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (probabilities[i] >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements
        // exceeds topp
        // O(k lg n0)
        float cumulative_prob = 0.0f;
        int last_idx = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulative_prob += probabilities[indices[i]];
            if (cumulative_prob > topp) {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= last_idx; i--) {
            cdf += probabilities[indices[i]];
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[last_idx]; // in case of rounding errors
    }

    /**
     * Uses greedy argmax sampling to return the index of the token with the highest
     * probability.
     *
     * @param probabilities
     *            The array of probabilities for each token.
     * @param n
     *            The number of tokens.
     * @return The index of the token with the highest probability.
     */
    static int sample_argmax(float[] probabilities, int n) {
        // return the index that has the highest probability
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < n; i++) {
            if (probabilities[i] > max_p) {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    /**
     * Samples a token based on the given logits and sampler parameters.
     *
     * @param sampler
     *            The sampler configuration containing temperature and other
     *            hyperparameters.
     * @param logits
     *            The array of logits representing the unnormalized probability
     *            distribution.
     * @return The sampled token index.
     */
    static int sample(Sampler sampler, float[] logits) {
        int next;
        if (sampler.temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            next = sample_argmax(logits, sampler.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q = 0; q < sampler.vocab_size; q++) {
                logits[q] /= sampler.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            InferenceEngine.softmax(logits, 0, sampler.vocab_size);
            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = sampler.random_f32();
            // we sample from this distribution to get the next token
            if (sampler.topp <= 0 || sampler.topp >= 1) {
                // simply sample from the predicted probability distribution
                next = sample_mult(logits, sampler.vocab_size, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
            }
        }
        return next;
    }

    /**
     * Generates a sequence of tokens using the provided Transformer model,
     * Tokenizer, Sampler, and input prompt.
     *
     * @param transformer
     *            The Transformer model used for sequence generation.
     * @param tokenizer
     *            The Tokenizer for encoding and decoding tokens.
     * @param sampler
     *            The Sampler for sampling the next token based on logits.
     * @param prompt
     *            The input prompt to start the generation process.
     * @param steps
     *            The maximum number of steps (tokens) to generate.
     */
    static void generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String prompt, int steps) {
        String empty_prompt = "";
        if (prompt == null) {
            prompt = empty_prompt;
        }

        // encode the (string) prompt into tokens sequence
        int num_prompt_tokens = 0; // the total number of prompt tokens
        int[] prompt_tokens = new int[prompt.length() * 2 + 3]; // +3 for '\0', ?BOS, ?EOS
        num_prompt_tokens = encode(tokenizer, prompt, true, false, prompt_tokens);
        if (num_prompt_tokens < 1) {
            System.err.println("something is wrong, expected at least 1 prompt token");
            System.exit(1);
        }

        TornadoExecutionPlan tornadoExecutionPlan = createTornadoExecutionPlan(transformer);

        long start = 0; // used to time our code, only initialized after first iteration
        int next; // will store the next token in the sequence
        int token = prompt_tokens[0]; // kick off with the first token in the prompt
        int pos = 0; // position in the sequence
        while (pos < steps) {
            // forward the transformer to get logits for the next token
            float[] logits = InferenceEngine.forward(transformer, token, pos, tornadoExecutionPlan);

            // Advance the state machine
            next = (pos < num_prompt_tokens - 1) ? prompt_tokens[pos + 1] : sample(sampler, logits);
            pos++;

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if (next == 1) {
                break;
            }

            // print the token as string, decode it with the Tokenizer object
            String piece = decode(tokenizer, token, next);
            safe_printf(piece);

            System.out.flush();
            token = next;

            // init the timer here because the first iteration can be slower
            if (start == 0) {
                start = time_in_ms();
            }
        }

        System.out.println();

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1) {
            long end = time_in_ms();
            System.err.printf("\nachieved tok/s: %f\n", (pos - 1) / (double) (end - start) * 1000);
        }
    }

    // ============= Chat =============

    /**
     * Reads a line from the stdin, up but not including the newline character.
     *
     * @param guide
     *            The prompt to display before reading the input.
     * @return The line read from the standard input.
     */
    static String read_stdin(String guide) {
        // read a line from stdin, up to but not including the newline character.
        System.out.print(guide);
        Scanner scanner = new Scanner(System.in);
        if (scanner.hasNextLine()) {
            return scanner.nextLine();
        }
        return null;
    }

    /**
     * Simulates a chat conversation with the Transformer model using given prompts
     * and system input. The conversation alternates between user and assistant
     * turns until the specified number of steps is reached.
     *
     * @param transformer
     *            The Transformer model used for generating responses.
     * @param tokenizer
     *            The Tokenizer for encoding and decoding text.
     * @param sampler
     *            The Sampler for sampling tokens based on logits.
     * @param cli_user_prompt
     *            The optional user prompt provided via the command line.
     * @param cli_system_prompt
     *            The optional system prompt provided via the command line.
     * @param steps
     *            The maximum number of conversation steps.
     */
    static void chat(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String cli_user_prompt, String cli_system_prompt, int steps) {

        // buffers for reading the system prompt and user prompt from stdin
        String system_prompt = null;
        String user_prompt = null;
        String rendered_prompt = null;
        int num_prompt_tokens = 0;
        int[] prompt_tokens = new int[512];
        int user_idx = 0;

        // start the main loop
        boolean user_turn = true; // user starts
        int next = 0; // will store the next token in the sequence
        int token = 0; // stores the current token to feed into the transformer
        int pos = 0; // position in the sequence
        while (pos < steps) {

            // when it is the user's turn to contribute tokens to the dialog...
            if (user_turn) {
                // get the (optional) system prompt at position 0
                if (pos == 0) {
                    // at position 0, the user can also contribute a system prompt
                    if (cli_system_prompt == null) {
                        // system prompt was not passed in, attempt to get it from stdin
                        system_prompt = read_stdin("Enter system prompt (optional): ");
                    } else {
                        // system prompt was passed in, use it
                        system_prompt = cli_system_prompt;
                    }
                }
                // get the user prompt
                if (pos == 0 && cli_user_prompt != null) {
                    // user prompt for position 0 was passed in, use it
                    user_prompt = cli_user_prompt;
                } else {
                    // otherwise get user prompt from stdin
                    user_prompt = read_stdin("User: ");
                }
                // render user/system prompts into the Llama 2 Chat schema
                if (pos == 0 && system_prompt.isEmpty()) {
                    String system_template = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                    rendered_prompt = system_template.formatted(system_prompt, user_prompt);
                } else {
                    String user_template = "[INST] %s [/INST]";
                    rendered_prompt = user_template.formatted(user_prompt);
                }
                // encode the rendered prompt into tokens
                num_prompt_tokens = encode(tokenizer, rendered_prompt, true, false, prompt_tokens);
                user_idx = 0; // reset the user index
                user_turn = false;
                System.out.print("Assistant: ");
            }

            // determine the token to pass into the transformer next
            if (user_idx < num_prompt_tokens) {
                // if we are still processing the input prompt, force the next prompt token
                token = prompt_tokens[user_idx++];
            } else {
                // otherwise use the next token sampled from previous turn
                token = next;
            }
            // EOS (=2) token ends the Assistant turn
            if (token == 2) {
                user_turn = true;
            }

            // forward the transformer to get logits for the next token
            float[] logits = InferenceEngine.forward(transformer, token, pos, null);
            next = sample(sampler, logits);
            pos++;

            if (user_idx >= num_prompt_tokens && next != 2) {
                // the Assistant is responding, so print its output
                String piece = decode(tokenizer, token, next);
                safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
                System.out.flush();
            }
            if (next == 2) {
                System.out.println();
            }
        }
        System.out.println();
    }

    /**
     * Prints error messages and usage examples.
     */
    static void error_usage() {
        System.err.println("Usage:   java Llama2 <checkpoint> [options]");
        System.err.println("Example: java Lamma2 model.bin -n 256 -i \"Once upon a time\"");
        System.err.println("Options:");
        System.err.println("  -t <float>  temperature in [0,inf], default 1.0");
        System.err.println("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9");
        System.err.println("  -s <int>    random seed, default time(NULL)");
        System.err.println("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
        System.err.println("  -i <string> input prompt");
        System.err.println("  -z <string> optional path to custom tokenizer");
        System.err.println("  -m <string> mode: generate|chat, default: generate");
        System.err.println("  -y <string> (optional) system prompt in chat mode");
        System.exit(1);
    }

    public static void main(String[] args) throws IOException {
        // default parameters
        String checkpoint_path = null; // e.g. out/model.bin
        String tokenizer_path = "tokenizer.bin";
        float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float topp = 0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
        long rng_seed = 0; // seed rng with time by default
        int steps = 256; // max number of steps to run for, 0: use seq_len
        String prompt = null; // prompt string
        String mode = "generate"; // generate|chat
        String system_prompt = null; // the (optional) system prompt to use in chat mode

        // parsing of arguments to override the defaults above from the command
        if (args.length >= 1) {
            checkpoint_path = args[0];
        } else {
            error_usage();
        }
        for (int i = 1; i < args.length; i += 2) {
            // do some basic validation
            if (i + 1 >= args.length) {
                error_usage();
            } // must have arg after flag
            if (args[i].charAt(0) != '-') {
                error_usage();
            } // must start with dash
            if (args[i].length() != 2) {
                error_usage();
            } // must be -x (one dash, one letter)
              // read in the args
            switch (args[i].charAt(1)) {
                case 't' -> temperature = Float.parseFloat(args[i + 1]);
                case 'p' -> topp = Float.parseFloat(args[i + 1]);
                case 's' -> rng_seed = Integer.parseInt(args[i + 1]);
                case 'n' -> steps = Integer.parseInt(args[i + 1]);
                case 'i' -> prompt = args[i + 1];
                case 'z' -> tokenizer_path = args[i + 1];
                case 'm' -> mode = args[i + 1];
                case 'y' -> system_prompt = args[i + 1];
                default -> error_usage();
            }
        }

        // parameter validation/overrides
        if (rng_seed <= 0) {
            rng_seed = System.currentTimeMillis();
        }
        if (temperature < 0.0) {
            temperature = 0.0f;
        }
        if (topp < 0.0 || 1.0 < topp) {
            topp = 0.9f;
        }
        if (steps <= 0) {
            steps = 0;
        }

        // build the Transformer via the model .bin file
        Transformer transformer = new Transformer(checkpoint_path);
        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len; // ovrerride to ~max length
        }

        // build the Tokenizer via the tokenizer .bin file

        Tokenizer tokenizer = new Tokenizer(tokenizer_path, transformer.config.vocab_size);

        // build the Sampler
        Sampler sampler = new Sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

        // run!
        switch (mode) {
            case "generate" -> generate(transformer, tokenizer, sampler, prompt, steps);
            case "chat" -> chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
            default -> {
                System.err.println("unknown mode: " + mode);
                error_usage();
            }
        }
    }

}
