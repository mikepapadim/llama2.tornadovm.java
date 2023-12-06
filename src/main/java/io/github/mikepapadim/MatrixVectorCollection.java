package io.github.mikepapadim;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat16;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat8;
import uk.ac.manchester.tornado.api.types.vectors.Float16;
import uk.ac.manchester.tornado.api.types.vectors.Float4;
import uk.ac.manchester.tornado.api.types.vectors.Float8;

/**
 * This class contains a set of Matrix and Vector multiplication methods implemented using TornadoVM's API.
 */
public class MatrixVectorCollection {

    private static final int TS = 4;

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It uses Graal's auto-vectorization for the matrix multiplication. However, if the
     * {@link #Llama2.USE_VECTOR_API} flag is set to true, it employs explicit vectorization
     * using the Vector API for further optimization.
     *
     * @param xout The output vector of the matrix multiplication.
     * @param x    The input vector to be multiplied with the weight matrix.
     * @param w    The weight matrix represented as a {@link FloatBuffer}.
     * @param n    The number of columns in the weight matrix and the size of the input vector.
     * @param d    The number of rows in the weight matrix and the size of the output vector.
     */
    static void matmul(float[] xout, float[] x, FloatBuffer w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        MemorySegment wSegment = MemorySegment.ofBuffer(w);
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;
            if (Llama2.USE_VECTOR_API) {
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

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * Is uses parallel streams for optimization.
     *
     * @param xout The output vector of the matrix multiplication.
     * @param x    The input vector to be multiplied with the weight matrix.
     * @param w    The weight matrix represented as a {@link FloatBuffer}.
     * @param n    The number of columns in the weight matrix and the size of the input vector.
     * @param d    The number of rows in the weight matrix and the size of the output vector.
     */
    static void matrixVectorMultiply(float[] xout, float[] x, FloatBuffer w, int n, int d) {
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        });
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It uses the @Parallel annotation of TornadoVM to signify to the compiler that the loop is data
     * parallel. The weight matrix w is represented by a TornadoVM {@link FloatArray}, which resides off-heap.
     *
     * @param xout The output vector of the matrix multiplication.
     * @param x    The input vector to be multiplied with the weight matrix.
     * @param w    The weight matrix represented as a {@link FloatArray}.
     * @param n    The number of columns in the weight matrix and the size of the input vector.
     * @param d    The number of rows in the weight matrix and the size of the output vector.
     */
    static void matrixVectorSimple(float[] xout, float[] x, FloatArray w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        }
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It uses the @Parallel annotation of TornadoVM to signify to the compiler that the loop is data
     * parallel. The weight matrix w and the input vector x are TornadoVM {@link VectorFloat16} vectors,
     * and reside off-heap.
     *
     * @param xout The output vector of the matrix multiplication.
     * @param x    The input vector to be multiplied with the weight matrix, represented as a {@link VectorFloat16}.
     * @param w    The weight matrix represented as a {@link VectorFloat16}.
     * @param n    The number of columns in the weight matrix and the size of the input vector.
     * @param d    The number of rows in the weight matrix and the size of the output vector.
     */
    static void matrixVectorFloat16(float[] xout, VectorFloat16 x, VectorFloat16 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            int vectorLaneWidth = x.vectorWidth();
            for (int j = 0; j < n; j += vectorLaneWidth) {
                Float16 xv16 = x.get(j / vectorLaneWidth);
                Float16 wv16 = w.get(i * (n / vectorLaneWidth) + j / vectorLaneWidth);
                val += Float16.dot(wv16, xv16);
            }
            xout[i] = val;
        }
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It uses the @Parallel annotation of TornadoVM to signify to the compiler that the loop is data
     * parallel. The weight matrix w and the input vector x are TornadoVM {@link VectorFloat8} vectors,
     * and reside off-heap.
     *
     * @param xout The output vector of the matrix multiplication.
     * @param x    The input vector to be multiplied with the weight matrix, represented as a {@link VectorFloat8}.
     * @param w    The weight matrix represented as a {@link VectorFloat8}.
     * @param n    The number of columns in the weight matrix and the size of the input vector.
     * @param d    The number of rows in the weight matrix and the size of the output vector.
     */
    static void matrixVectorFloat8(float[] xout, VectorFloat8 x, VectorFloat8 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            int vectorLaneWidth = x.vectorWidth();
            for (int j = 0; j < n; j += vectorLaneWidth) {
                Float8 xv8 = x.get(j / vectorLaneWidth);
                Float8 wv8 = w.get(i * (n / vectorLaneWidth) + j / vectorLaneWidth);
                val += Float8.dot(wv8, xv8);
            }
            xout[i] = val;
        }
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It uses the @Parallel annotation of TornadoVM to signify to the compiler that the loop is data
     * parallel. The weight matrix w and the input vector x are TornadoVM {@link VectorFloat4} vectors,
     * and reside off-heap.
     *
     * @param xout The output vector of the matrix multiplication.
     * @param x    The input vector to be multiplied with the weight matrix, represented as a {@link VectorFloat4}.
     * @param w    The weight matrix represented as a {@link VectorFloat4}.
     * @param n    The number of columns in the weight matrix and the size of the input vector.
     * @param d    The number of rows in the weight matrix and the size of the output vector.
     */
    static void matrixVectorFloat4(float[] xout, VectorFloat4 x, VectorFloat4 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            int vectorLaneWidth = x.vectorWidth();
            for (int j = 0; j < n; j += vectorLaneWidth) {
                Float4 xv4 = x.get(j / vectorLaneWidth);
                Float4 wv4 = w.get(i * (n / vectorLaneWidth) + j / vectorLaneWidth);
                val += Float4.dot(wv4, xv4);
            }
            xout[i] = val;
        }
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It is implemented using TornadoVM's Kernel API.
     * The weight matrix w is represented by a TornadoVM {@link FloatArray}, which resides off-heap.
     *
     * @param xout    The output vector of the matrix multiplication.
     * @param x       The input vector to be multiplied with the weight matrix.
     * @param w       The weight matrix represented as a {@link FloatArray}.
     * @param n       The number of columns in the weight matrix and the size of the input vector.
     * @param context The kernel context object is used to access low-level parallel programming components
     *                (e.g. global and local thread id, etc.).
     */
    static void matrixVectorSimpleWithContext(float[] xout, float[] x, FloatArray w, int n, KernelContext context) {
        int idx = context.globalIdx;
        float val = 0f;
        for (int j = 0; j < n; j++) {
            val += w.get(idx * n + j) * x[j];
        }
        xout[idx] = val;
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It is implemented using TornadoVM's Kernel API and uses a local array for optimization.
     * The weight matrix w is represented by a TornadoVM {@link FloatArray}, which resides off-heap.
     *
     * @param xout    The output vector of the matrix multiplication.
     * @param x       The input vector to be multiplied with the weight matrix.
     * @param w       The weight matrix represented as a {@link FloatArray}.
     * @param n       The number of columns in the weight matrix and the size of the input vector.
     * @param context The kernel context object is used to access low-level parallel programming components
     *                (e.g. global and local thread id, etc.).
     */
    static void matrixVectorOptimizedWithContext(float[] xout, float[] x, FloatArray w, int n, KernelContext context) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;

        float[] wSub = context.allocateFloatLocalArray(TS); // TS is the tile size

        float val = 0.0f;

        // Loop over tiles
        int numTiles = n / TS;
        for (int tileIndex = 0; tileIndex < numTiles; tileIndex++) {
            int globalCol = TS * tileIndex + localIdx;

            // Load one tile of the weight matrix into local memory
            wSub[localIdx] = w.get(globalIdx * n + globalCol);

            // Synchronize to make sure the tile is loaded
            context.localBarrier();

            // Perform the computation for a single tile
            for (int k = 0; k < TS; k++) {
                val += wSub[k] * x[TS * tileIndex + k];
            }

            // Synchronize before loading the next tile
            context.localBarrier();
        }

        // Store the final result in xout
        xout[globalIdx] = val;
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It is implemented using TornadoVM's Kernel API.
     * The weight matrix w and the input vector x are TornadoVM {@link VectorFloat16} vectors, and reside off-heap.
     *
     * @param xout    The output vector of the matrix multiplication.
     * @param x       The input vector to be multiplied with the weight matrix, represented as a {@link VectorFloat16}.
     * @param w       The weight matrix represented as a {@link VectorFloat16}.
     * @param n       The number of columns in the weight matrix and the size of the input vector.
     * @param context The kernel context object is used to access low-level parallel programming components
     *                (e.g. global and local thread id, etc.).
     */
    static void matrixVectorFloat16withContext(float[] xout, VectorFloat16 x, VectorFloat16 w, int n, KernelContext context) {
        int idx = context.globalIdx;
        float val = 0f;
        int vectorLaneWidth = x.vectorWidth();
        for (int j = 0; j < n; j += vectorLaneWidth) {
            Float16 xv16 = x.get(j / vectorLaneWidth);
            Float16 wv16 = w.get(idx * (n / vectorLaneWidth) + j / vectorLaneWidth);
            val += Float16.dot(wv16, xv16);
        }
        xout[idx] = val;
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It is implemented using TornadoVM's Kernel API.
     * The weight matrix w and the input vector x are TornadoVM {@link VectorFloat8} vectors, and reside off-heap.
     *
     * @param xout    The output vector of the matrix multiplication.
     * @param x       The input vector to be multiplied with the weight matrix, represented as a {@link VectorFloat8}.
     * @param w       The weight matrix represented as a {@link VectorFloat8}.
     * @param n       The number of columns in the weight matrix and the size of the input vector.
     * @param context The kernel context object is used to access low-level parallel programming components
     *                (e.g. global and local thread id, etc.).
     */
    static void matrixVectorFloat8KwithContext(float[] xout, VectorFloat8 x, VectorFloat8 w, int n, KernelContext context) {
        int idx = context.globalIdx;
        int vectorLaneWidth = w.vectorWidth();
        float val = 0f;
        for (int j = 0; j < n; j += vectorLaneWidth) {
            Float8 xv8 = x.get(j / vectorLaneWidth);
            Float8 wv8 = w.get(idx * (n / vectorLaneWidth) + j / vectorLaneWidth);
            val += Float8.dot(wv8, xv8);
        }
        xout[idx] = val;
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input vector (x).
     * It is implemented using TornadoVM's Kernel API.
     * The weight matrix w and the input vector x are TornadoVM {@link VectorFloat4} vectors, and reside off-heap.
     *
     * @param xout    The output vector of the matrix multiplication.
     * @param x       The input vector to be multiplied with the weight matrix, represented as a {@link VectorFloat4}.
     * @param w       The weight matrix represented as a {@link VectorFloat4}.
     * @param n       The number of columns in the weight matrix and the size of the input vector.
     * @param context The kernel context object is used to access low-level parallel programming components
     *                (e.g. global and local thread id, etc.).
     */
    static void matrixVectorFloat4withContext(float[] xout, VectorFloat4 x, VectorFloat4 w, int n, KernelContext context) {
        int idx = context.globalIdx;
        float val = 0f;
        int vectorLaneWidth = x.vectorWidth();
        for (int j = 0; j < n; j += vectorLaneWidth) {
            Float4 xv4 = x.get(j / vectorLaneWidth);
            Float4 wv4 = w.get(idx * (n / vectorLaneWidth) + j / vectorLaneWidth);
            val += Float4.dot(wv4, xv4);
        }
        xout[idx] = val;
    }

}
