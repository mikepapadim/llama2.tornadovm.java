# A Llama2 implementation accelerated with TornadoVM

This repository provides an implementation of [llama2.java](https://github.com/mukel/llama2.java), extended to use the Vector API and [TornadoVM](https://github.com/beehive-lab/TornadoVM) for acceleration.
Additionally, developers can optionally run with three different vector types, Vector4, Vector8 or Vector16, optimized by TornadoVM.

## Prerequisites
* **Java 21+**: This is essential as the project uses the [Project Panama](https://openjdk.org/projects/panama/) for native memory allocation. 
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

And finally compile the project by running this script:

```bash
./compile.sh
```

## Execution
### Token files
Just like the original java implementation, the program requires a .bin file with an input model. 
Models of various sizes can be downloaded from [here](https://huggingface.co/karpathy/tinyllamas/tree/main).
### How to run
The repository contains a `run.sh` script for running. This script takes the following arguments:
* The TornadoVM workgroup size (optional)
* The TornadoVM Vector mode (optional)
* The .bin model file

```bash
// Run with just the model 
./run.sh stories15M.bin not working 
// Run with the workgroup size and the model
./run.sh -n 128 stories15M.bin
// Run with the workgroup size, the [VectorFloat4|VectorFloat8|VectorFloat16] types and the model
./run.sh -n 128 -v -Dllama2.VectorFloat4=true stories15M.bin
// Run with the [VectorFloat4|VectorFloat8|VectorFloat16] types and the model
./run.sh -v -Dllama2.VectorFloat4=true stories15M.bin

```

## Performance

Quick numbers on an AMD Ryzen 3950X 64GB, Arch Linux.  
`llama2.java` executed on OpenJDK 20.0.2+9.  
To make things fair w.r.t. to vectorization, the Java version has a matmul implementation using the [Vector API](https://openjdk.org/jeps/448).  
In these measurements the JVM is warmed up enough to reach peak tokens/s.  
On GraalVM, please note that the Graal compiler doesn't support the Vector API yet, to avoid unexpected performance degradation, run with `-Dllama2.VectorAPI=false`.

****Notes**  
*The numbers below were collected using aggressive (gcc) compiler flags e.g. regular `gcc -O2 ...` wouldn't be as fast.*

### Single-threaded

`llama2.c` compiled with `gcc -Ofast -march=native run.c -lm -o run -march=native`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=0`

| Model | Tokens per second | Speedup vs. llama2.c | Implementation |  
| ------|------------------ | -------------------- | -------------- | 
| stories15M.bin  |   363 |  1.0 | llama2.c    |
| stories15M.bin  |   237 | 0.65 | llama2.java |
| stories110M.bin | 51.71 |  1.0 | llama2.c    |
| stories110M.bin | 42.20 | 0.81 | llama2.java |
| llama2_7B.bin   |  0.92 |  1.0 | llama2.c    |
| llama2_7B.bin   |  0.88 | 0.95 | llama2.java |

### Multi-threaded

`llama2.c` compiled with `gcc -Ofast -fopenmp -march=native run.c -lm -o run -march=native`  
`llama2.c` executed with `OMP_NUM_THREADS=8`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=8`  

| Model | Tokens per second | Speedup vs. llama2.c | Implementation |  
| ------|------------------ | -------------------- | -------------- |
|  stories15M.bin |  1233 |  1.0 | llama2.c    |
|  stories15M.bin |   438 | 0.35 | llama2.java |
| stories110M.bin |    90 |  1.0 | llama2.c    |
| stories110M.bin |    80 | 0.88 | llama2.java |
|   llama2_7B.bin |  1.68 |  1.0 | llama2.c    |
|   llama2_7B.bin |  1.65 | 0.98 | llama2.java |

****Notes**  
*In `stories15M.bin`, the C version shows a huge speedup, very likely a cache effect, this is considered an outlier.
Running with 16/32 threads may actually cause a slowdown; the performance is, in most cases, U-shaped w.r.t to the # of threads.
With that many threads, vectorization does not give any advantage, since throughput is limited by memory bandwidth.*

Performance is already comparable to the original C code, bar vectorization, even if the Java code has not been optimized yet.

## License

MIT
