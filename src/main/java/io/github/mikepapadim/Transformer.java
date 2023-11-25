package io.github.mikepapadim;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class Transformer {
    final Config config; // the hyperparameters of the architecture (the blueprint)
    final Weights weights; // the weights of the model
    final RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    final Arena memoryArena; // scope of the memory mapping
    final MemorySegment data; // memory mapped data pointer
    final long file_size; // size of the checkpoint file in bytes

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