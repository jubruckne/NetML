using System.Runtime.InteropServices;

namespace NetML.ML;

public static unsafe class Accelerate {
    private const string AccelerateLibrary = "/System/Library/Frameworks/Accelerate.framework/Accelerate";

    [DllImport(AccelerateLibrary)]
    public static extern void cblas_saxpy(int n, float alpha, float[] x, int incx, float[] y, int incy);

    [DllImport(AccelerateLibrary)]
    public static extern void cblas_scopy(int n, float[] x, int incx, float[] y, int incy);

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_vadd(float* A, int IA, float* B, int IB, float* C, int IC, uint N);

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_vsub(float* A, int IA, float* B, int IB, float* C, int IC, uint N);

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_vmul(float* A, int IA, float* B, int IB, float* C, int IC, uint N);

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_vma(float[] A,
                                       int IA,
                                       float[] B,
                                       int IB,
                                       float[] C,
                                       int IC,
                                       float[] D,
                                       int ID,
                                       uint N
    );

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_vsum(float[] A, int IA, float[] C, uint N);

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_maxv(float[] A, int IA, float[] C, uint N);

    [DllImport(AccelerateLibrary)]
    public static extern void vDSP_mmul(float* A,
                                        int IA,
                                        float* B,
                                        int IB,
                                        float* C,
                                        int IC,
                                        uint M,
                                        uint N,
                                        uint P
    );

    [DllImport(AccelerateLibrary)]
    public static extern unsafe void cblas_sgemv(int order,
                                                 int trans,
                                                 int m,
                                                 int n,
                                                 float alpha,
                                                 float* a,
                                                 int lda,
                                                 float* x,
                                                 int incx,
                                                 float beta,
                                                 float* y,
                                                 int incy
    );

    [DllImport(AccelerateLibrary)]
    public static extern unsafe void vDSP_vsmsb(float* a,
                                                int aStride,
                                                float* b,
                                                int bStride,
                                                float* c,
                                                int cStride,
                                                float* d,
                                                int dStride,
                                                uint count
    );

    [DllImport(AccelerateLibrary)]
    public static extern unsafe void vDSP_vsmul(float* a,
                                                int aStride,
                                                float* b,
                                                float* c,
                                                int cStride,
                                                uint count
    );

    [DllImport(AccelerateLibrary)]
    public static extern unsafe void vDSP_vsma(float* a,
                                               int aStride,
                                               float* b,
                                               float* c,
                                               int cStride,
                                               float* d,
                                               int dStride,
                                               uint count
    );

    [DllImport(AccelerateLibrary)]
    public static extern unsafe void vDSP_vsdiv(float* a, int aStride, float* b, float* c, int cStride, uint count);

    [DllImport(AccelerateLibrary)]
    public static extern unsafe void cblas_sgemm(int order,
                                                 int transA,
                                                 int transB,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 float alpha,
                                                 float* a,
                                                 int lda,
                                                 float* b,
                                                 int ldb,
                                                 float beta,
                                                 float* c,
                                                 int ldc
    );

}