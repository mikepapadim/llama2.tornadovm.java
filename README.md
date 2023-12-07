# An extension of the Llama2.java implementation, accelerated with GPUs by using TornadoVM

This repository provides an implementation of [llama2.java](https://github.com/mukel/llama2.java), extended to use [TornadoVM](https://github.com/beehive-lab/TornadoVM) for GPU acceleration of the original Java code.
Additionally, developers can optionally run with three different vector types, Vector4, Vector8 or Vector16, optimized by TornadoVM.

## Prerequisites
* **JDK 21+**: This is essential as the project uses the [Project Panama](https://openjdk.org/projects/panama/) for native memory allocation. 
* **TornadoVM**: Detailed installation instructions can be found [here](https://tornadovm.readthedocs.io/en/latest/installation.html).  

## Build
The `set_paths.sh` file provides a template with all the paths that need to be set up for the compilation and execution.
From this template, the paths that need to be set are: 
* **$JAVA_HOME**, with the path to JDK 21
* **$TORNADO_ROOT**, with the path to the TornadoVM installation  
* **$LLAMA_ROOT**, with the path of this project.

After the `set_paths.sh` file has been configured with the correct paths, run:

```bash
./set_paths.sh  
```

And finally, compile the project by running this script:

```bash
./compile.sh
```

## Execution
### Token files
Just like the original Java implementation, the program requires a `tokenizer.bin` file and the input models available in the TinyLlamas. 
```bash
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories100M.bin
```
### How to run
The repository contains a `run.sh` script for running. This script takes the following arguments:
* The TornadoVM workgroup size (optional)
* The TornadoVM Vector mode (optional)
* The .bin model file

Additionally, the script can take an optional that enables the execution of the program in pure Java, without TornadoVM.

```bash
// Run with just the model 
./run.sh stories15M.bin 
// Run with the workgroup size and the model
./run.sh -n 128 stories15M.bin
// Run with the workgroup size, the [VectorFloat4|VectorFloat8|VectorFloat16] types and the model
./run.sh -n 128 -v -Dllama2.VectorFloat4=true stories15M.bin
// Run with the [VectorFloat4|VectorFloat8|VectorFloat16] types and the model
./run.sh -v -Dllama2.VectorFloat4=true stories15M.bin
// Run in pure Java, without TornadoVM
./run.sh -j java
```

## Performance

| Component  | Specification                              |
|------------|--------------------------------------------|
| CPU        | 13th Gen Intel® Core i7-13700 × 24 threads |
| GPU        | NVIDIA GeForce RTX 3070                    |
| OS         | Pop!_OS Linux                              |
| JDK        | OpenJDK 21+35-2513                         |
| TornadoVM  | v1.0                                       |

**Test Objective: Synergy Between Vector API, Panama  and TornadoVM**

This test aims to illustrate the collaborative efficiency gained by integrating Vector API, employing off-heap memory types via MemorySegments for read-only weights, and TornadoVM. Following the profiling of the original Java implementation, optimization was directed at offloading only the last matrix vector computation to the GPU through TornadoVM.

To ensure unbiased and reliable performance evaluation, the test will be executed over more than 100 iterations. This extended duration allows the Java Virtual Machine (JVM) to reach a warm-up state, ensuring stability in performance measurements.

### Multi-threaded
 
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=24`  
We record in the following table the maximum of tokens per second achieved after warm-up.

| Model | Tokens per second | Speedup vs. llama2.java | Implementation |  
| ------|------------------ | -------------------- | -------------- |
|  stories15M.bin |  718 |  **1.15x** | llama2TornadoVM.java |
|  stories15M.bin |   626 | 1.0 | llama2.java |
| stories42M.bin |    326 |  **1.16x** | llama2TornadoVM.java    |
| stories420M.bin |   281 | 1.0 | llama2.java |
| stories110M.bin |  137 |  **1.09x** | llama2TornadoVM.java    |
| stories110M.bin |  126 | 1.0 | llama2.java |

----------------------------------------------

## License

MIT
