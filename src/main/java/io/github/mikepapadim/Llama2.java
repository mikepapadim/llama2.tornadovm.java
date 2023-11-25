package io.github.mikepapadim;

///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 20
//COMPILE_OPTIONS --enable-preview -source 20 --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules=jdk.incubator.vector
//NATIVE_OPTIONS  --enable-preview --add-modules=jdk.incubator.vector --initialize-at-build-time=Llama2 -Dllama2.VectorAPI=false

/* Inference for Llama-2 Transformer model in pure Java */

// ----------------------------------------------------------------------------
// Transformer model

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat8;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
import uk.ac.manchester.tornado.api.types.vectors.Float8;

class Llama2 {

    // ----------------------------------------------------------------------------
    // neural net blocks; the dynamics of the Transformer

    static final boolean USE_VECTOR_API = "true".equalsIgnoreCase(System.getProperty("llama2.VectorAPI", "true"));
    static final boolean USE_VECTORFLOAT8 = "true".equalsIgnoreCase(System.getProperty("llama2.VectorFloat8", "false"));
    static final boolean USE_VECTORFLOAT4 = "true".equalsIgnoreCase(System.getProperty("llama2.VectorFloat4", "false"));

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

    static void matmul(float[] xout, float[] x, FloatBuffer w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        MemorySegment wSegment = MemorySegment.ofBuffer(w);
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;
            if (USE_VECTOR_API) {
                VectorSpecies<Float> species = FloatVector.SPECIES_256;
                FloatVector sum0 = FloatVector.zero(species);
                FloatVector sum1 = FloatVector.zero(species);
                FloatVector sum2 = FloatVector.zero(species);
                FloatVector sum3 = FloatVector.zero(species);
                int width = species.length();
                int upperBound = n - n % (4 * width);
                for (; j < upperBound; j += 4 * width) {
                    var wj0 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 0 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj1 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 1 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj2 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 2 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj3 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 3 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var xj0 = FloatVector.fromArray(species, x, j + 0 * width);
                    var xj1 = FloatVector.fromArray(species, x, j + 1 * width);
                    var xj2 = FloatVector.fromArray(species, x, j + 2 * width);
                    var xj3 = FloatVector.fromArray(species, x, j + 3 * width);
                    sum0 = wj0.fma(xj0, sum0);
                    sum1 = wj1.fma(xj1, sum1);
                    sum2 = wj2.fma(xj2, sum2);
                    sum3 = wj3.fma(xj3, sum3);
                }
                val = sum0.add(sum1).add(sum2).add(sum3).reduceLanes(VectorOperators.ADD);
            }

            // Graal's auto-vectorization.
            int upperBound = n & ~3;
            float[] sum = new float[4];
            for (; j < upperBound; j += sum.length) {
                sum[0] += w.get(i * n + j + 0) * x[j + 0];
                sum[1] += w.get(i * n + j + 1) * x[j + 1];
                sum[2] += w.get(i * n + j + 2) * x[j + 2];
                sum[3] += w.get(i * n + j + 3) * x[j + 3];
            }
            val += sum[0] + sum[1] + sum[2] + sum[3];

            for (; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        });
    }

    static void matrixVectorSimple(float[] xout, float[] x, FloatArray w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        }
    }

    static void matrixVectorFloat8(float[] xout, VectorFloat8 x, VectorFloat8 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;

            for (int j = 0; j < n; j += 8) {
                Float8 xv8 = x.get(j / 8);
                Float8 wv8 = w.get(i * (n / 8) + j / 8);

                val += Float8.dot(wv8, xv8);

            }

            xout[i] = val;
        }
    }

    static void matrixVectorFloat4(float[] xout, VectorFloat4 x, VectorFloat4 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;

            for (int j = 0; j < n; j += 4) {
                Float4 xv4 = x.get(j / 4);
                Float4 wv4 = w.get(i * (n / 4) + j / 4);
                val += Float4.dot(wv4, xv4);
            }

            xout[i] = val;
        }
    }

    static float[] forward(Transformer transformer, int token, int pos, ArrayList<TornadoExecutionPlan> executionPlan) {
        // a few convenience variables
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
            matmul(s.q, s.xb, w.wq[l], dim, dim);
            matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
            matmul(s.v, s.xb, w.wv[l], dim, kv_dim);

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
            matmul(s.xb2, s.xb, w.wo[l], dim, dim);

            residualConnection(s.x, s.xb2, dim);

            // ffn rmsnorm
            rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.w1[l], dim, p.hidden_dim);
            matmul(s.hb2, s.xb, w.w3[l], dim, p.hidden_dim);

            fusedSiluEwiseMul(hidden_dim, s.hb, s.hb2);

            // final matmul to get the output of the ffn
            matmul(s.xb, s.hb, w.w2[l], p.hidden_dim, dim);

            residualConnection(s.x, s.xb, dim);
        }

        // final rmsnorm
        rmsnorm(s.x, s.x, w.rms_final_weight, dim);

        // convertToVectorFloat8(s.xV8, s.x);
        convertToVectorFloat4(s.xVectorFloat4, s.x);

        executionPlan.get(executionPlan.size() - 1).withDevice(TornadoExecutionPlan.getDevice(0, 0)).execute();

        return s.logits;
    }

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

    // SwiGLU non-linearity
    static void fusedSiluEwiseMul(int hidden_dim, float[] out, float[] hb2) {
        for (int i = 0; i < hidden_dim; i++) {
            float val = out[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + Math.exp(-val)));
            // elementwise multiply with w3(x)
            out[i] = val * hb2[i];
        }
    }

    static void residualConnection(float[] s, float[] xb2, int dim) {
        for (int i = 0; i < dim; i++) {
            s[i] = s[i] + xb2[i];
        }
    }
    // ----------------------------------------------------------------------------

    // The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

    static String decode(Tokenizer t, int prev_token, int token) {
        String piece = t.vocab[token];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace
        // (see PR #89)
        if (prev_token == 1 && piece.charAt(0) == ' ') {
            piece = piece.substring(1);
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        String prefix = "<0x";
        String suffix = ">";
        if (piece.length() == 6 && piece.startsWith(prefix) && piece.endsWith(suffix)) {
            String hex2 = piece.substring(prefix.length(), prefix.length() + 2);
            char ch = (char) Integer.parseInt(hex2, 16);
            // ok this token is a raw byte token, carefuly to only print printable chars or
            // whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            piece = Character.toString(ch);
        }
        return piece;
    }

    static void safe_printf(String piece) {
        // piece might be a raw byte token, and we only want to print printable chars or
        // whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (piece == null) {
            return;
        }
        if (piece.isEmpty()) {
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

    static int str_lookup(String str, Map<String, Integer> sorted_vocab) {
        // efficiently find the perfect match for str in vocab, return its index or -1
        // if not found
        return sorted_vocab.getOrDefault(str, -1);
    }

    static int encode(Tokenizer t, String text, boolean bos, boolean eos, int[] tokens) {
        // encode the string text (input) into an upper-bound preallocated tokens[]
        // array
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

        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
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

    // ----------------------------------------------------------------------------
    // utilities: time / rng

    static long time_in_ms() {
        // return time in milliseconds, for benchmarking the model speed
        return System.nanoTime() / 1_000_000;
    }

    // ----------------------------------------------------------------------------
    // generation loop

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

        Config p = transformer.config;
        Weights w = transformer.weights;
        RunState s = transformer.state;
        int dim = p.dim;

        TaskGraph taskGraph = new TaskGraph("s0").transferToDevice(DataTransferMode.EVERY_EXECUTION, s.xVectorFloat4).transferToDevice(DataTransferMode.FIRST_EXECUTION, w.weightInVectorFloat4)
                .task("t0", Llama2::matrixVectorFloat4, s.logits, s.xVectorFloat4, w.weightInVectorFloat4, dim, p.vocab_size)
                // .task("t0", Llama2::matmuxl2,s.logits, s.x, w.fa, dim, p.vocab_size)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, s.logits);

        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        ArrayList<TornadoExecutionPlan> te = new ArrayList<>();

        // for(int i = 0; i < p.n_layers; i++) {
        // TaskGraph taskGraph0 = new TaskGraph("sx" + i)
        // .transferToDevice(DataTransferMode.EVERY_EXECUTION, s.xb)
        // .transferToDevice(DataTransferMode.FIRST_EXECUTION,
        // w.weightsAsPrimitivesQ.get(i), w.weightsAsPrimitivesK.get(i),
        // w.weightsAsPrimitivesV.get(i))
        // .task("t1", Llama2::matmul2,s.q, s.xb, w.weightsAsPrimitivesQ.get(i), dim,
        // dim)
        // .task("t2", Llama2::matmul2,s.k, s.xb, w.weightsAsPrimitivesK.get(i), dim,
        // kv_dim)
        // .task("t3", Llama2::matmul2,s.v, s.xb, w.weightsAsPrimitivesV.get(i), dim,
        // kv_dim)
        // .transferToHost(DataTransferMode.EVERY_EXECUTION,s.q, s.k, s.v);
        // te.add(new TornadoExecutionPlan(taskGraph0.snapshot()));
        // }

        te.add(new TornadoExecutionPlan(taskGraph.snapshot()));
        // init tornado
        // start the main loop
        long start = 0; // used to time our code, only initialized after first iteration
        int next; // will store the next token in the sequence
        int token = prompt_tokens[0]; // kick off with the first token in the prompt
        int pos = 0; // position in the sequence
        while (pos < steps) {
            // forward the transformer to get logits for the next token
            float[] logits = forward(transformer, token, pos, te);

            // advance the state machine
            if (pos < num_prompt_tokens - 1) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos + 1];
            } else {
                // otherwise sample the next token from the logits
                next = sample(sampler, logits);
            }
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

    // ----------------------------------------------------------------------------
    // sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

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

    static int sample_mult(float[] probabilities, int n, float coin) {
        // sample index from probabilities (they must sum to 1!)
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

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

    static int sample_topp(float[] probabilities, int n, float topp, int[] indices, float coin) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()
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

    static int sample(Sampler sampler, float[] logits) {
        // sample the token given the logits and some hyperparameters
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
            softmax(logits, 0, sampler.vocab_size);
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

    static String read_stdin(String guide) {
        // read a line from stdin, up to but not including \n
        System.out.print(guide);
        Scanner scanner = new Scanner(System.in);
        if (scanner.hasNextLine()) {
            return scanner.nextLine();
        }
        return null;
    }

    // ----------------------------------------------------------------------------
    // chat loop
    // I manually inspected the tokens for a few chat conversations compared to
    // python reference and that seemed ok, but this was not thoroughly tested and
    // is not safely implemented, it's more a proof of concept atm.

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
        int prev_token;
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
            float[] logits = forward(transformer, token, pos, null);
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

    // ----------------------------------------------------------------------------

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

        // poor man's C argparse so we can override the defaults above from the command
        // line
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
