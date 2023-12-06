package io.github.mikepapadim;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * The Transformer class represents a neural network model with hyperparameters,
 * weights, and state information for performing forward passes.
 */
public class Transformer {

    /**
     * The hyperparameters of the architecture (the blueprint).
     */
    final Config config;

    /**
     * The weights of the model.
     */
    final Weights weights;

    /**
     * Buffers for the "wave" of activations in the forward pass.
     */
    final RunState state;

    /**
     * Scope of the memory mapping for proper memory cleanup.
     */
    final Arena memoryArena;

    /**
     * Memory-mapped data pointer containing the checkpoint file.
     */
    final MemorySegment data;

    /**
     * Size of the checkpoint file in bytes.
     */
    final long file_size;

    /**
     * Constructs a Transformer by loading the model checkpoint from a file.
     *
     * @param checkpointPath
     *            The path to the checkpoint file.
     * @throws IOException
     *             If an I/O error occurs while reading the checkpoint file.
     */
    public Transformer(String checkpointPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpointPath), StandardOpenOption.READ)) {
            this.file_size = fileChannel.size();
            this.memoryArena = Arena.ofAuto();
            this.data = memoryArena.allocate(this.file_size, 1);

            // Read the entire file into the MemorySegment
            fileChannel.read(this.data.asByteBuffer());

            int configSize = 7 * Integer.BYTES;
            // Read in the config header
            MemorySegment configSegment = this.data.asSlice(0, configSize);
            ByteBuffer configBuffer = configSegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            this.config = new Config(configBuffer);
            System.out.println(this.config);

            this.state = new RunState(this.config);

            // Move the position to the beginning of the weights data
            MemorySegment weightsSegment = this.data.asSlice(configSize);
            this.weights = new Weights(this.config, weightsSegment);
        }
    }
}
