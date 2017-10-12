#include "cudamatrix.h"

// C = alpha * A * B + beta * C
void cumatrixXmatrix (matrix *A, matrix *B, matrix *C, bool transA, bool transB, real alpha /* = 1.0 */, real beta /* = 0.0 */, int Cbias  /* =0 */)
{
    printf ("Error for calling function: cumatrixXmatrix!\n");
}

// A: layer0    B: neu0_ac      C: neu1_ac
void cumatrixXmatrix_fw (matrix *A, matrix *B, matrix *C, bool transA, bool transB, real alpha /* = 1.0 */, real beta /* = 0.0 */, int Bbias  /* =0 */, int Cbias, int chunksize)
{
    printf ("Error for calling function: cumatrixXmatrix_fw!\n");
}

void bperWordlayer (matrix *layers, matrix *srcer, matrix *tgter, int *curclass, int *classinfo, float alpha, float beta)
{
    printf ("Error for calling function: bperWordlayer!\n");
}

void bpupdateWordlayer (matrix *ac, matrix *er, matrix *layers, int *curclass, int *classinfo, float alpha, float beta)
{
    printf ("Error for calling function: bpupdateWordlayer!\n");
}
