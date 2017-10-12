#ifndef _CUDAMATRIX_H__
#define _CUDAMATRIX_H__
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper.h"
#include "head.h"
#include "DataType.h"

// current code should guranttee after call the function, GPU and CPU has the same value in the corresponding postion.
class matrix
{
private:
    real* host_data;
    real* dev_data;
    size_t nrows;
    size_t ncols;
    size_t size;
public:
    matrix ():host_data(NULL), dev_data(NULL), nrows(0), ncols(0)
    {}
    matrix (size_t nr, size_t nc)
    {
        nrows = nr;
        ncols = nc;
        size = sizeof(real) * ncols * nrows;
        host_data = (real *) malloc (size);
        dev_data = NULL;
    }
    ~matrix ()
    {
        if (host_data)
        {
            free (host_data);
            host_data = NULL;
        }
    }
    size_t Sizeof ()
    {
        return (nrows * ncols * sizeof(real));
    }
    size_t nelem ()
    {
        return (nrows * ncols);
    }
    // copy all data from CPU to GPU
    void assign ()
    {
        printf ("Error for calling function: assign!\n");
    }

    // assign value on both CPU and GPU for elem[i,j]
    void assignvalue(size_t i,size_t j, real v)
    {
        printf ("Error for calling function: assignvalue!\n");
    }
    // assign value on GPU
    void assigndevvalue (size_t i, size_t j, real v)
    {
        printf ("Error for calling function: assigndevvalue!\n");
    }
    // assign value in one column of matrix on GPU
    void assigndevcolumnvalue (size_t j, real v)
    {
        printf ("Error for calling function: assigndevcolumnvalue!\n");
    }
    // asign value on CPU
    void assignhostvalue (size_t i, size_t j, real v)
    {
        host_data[i+j*nrows] = v;
    }
    real addhostvalue (size_t i, size_t j, real v)
    {
        return (host_data[i+j*nrows] += v);
    }

    void assignneu0ac (int *prevwords, size_t mb, real v)
    {
        printf ("Error for calling function: assignneu0ac!\n");
    }

    // copy all data from GPU to CPU
    void fetch ()
    {
        printf ("Error for calling function: fetch!\n");
    }

    real fetchvalue (size_t i, size_t j)
    {
        printf ("Error for calling function: fetchvalue!\n");
        return 0;
    }
    real fetchhostvalue (size_t i, size_t j)
    {
        return host_data[i+j*nrows];
    }


    // get the word probabilities from top layer after forward, used for testNet and testNbest
    void fetchwordprobs (int *dev_curwords, size_t mb, real *wordprobs, int fulldict_size)
    {
        printf ("Error for calling function: fetchwordprobs!\n");
    }

    // ensure the value in GPU and CPU is the same.
    void checkCPUandGPUmem ()
    {
        printf ("Error for calling function: checkCPUandGPUmem!\n");
    }

    void setnrows (size_t nr)
    {
        nrows = nr;
    }
    void setncols (size_t nc)
    {
        ncols = nc;
    }
    size_t rows ()
    {
        return nrows;
    }
    size_t cols ()
    {
        return ncols;
    }
    void freemem ()
    {
        free (host_data);
        // cufree (dev_data);
        ncols = 0;
        nrows = 0;
        size = 0;
    }
    real& operator() (int i, int j) const
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        return host_data[i + j*nrows];
    }
    const real& operator() (int i, int j)
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        return host_data[i + j*nrows];
    }
    real* getdevdataptr ()
    {
        return dev_data;
    }
    real *getdevdataptr (int i, int j)
    {
        return &dev_data[i+j*nrows];
    }
    real* gethostdataptr ()
    {
        return host_data;
    }
    real *gethostdataptr(int i, int j)
    {
        return &host_data[i+j*nrows];
    }
    void assign (size_t i0, size_t j0, size_t nr, size_t nc, matrix* &other)
    {
        printf ("Error for calling function: assign!\n");
    }


    void getdevsubmatrix (matrix *other, size_t i0, size_t j0, size_t nr, size_t nc)
    {
        printf ("Error for calling function: getdevsubmatrix!\n");
    }

    // assign matrix from element from another matrix (other), both GPU and CPU
    // TODO: write GPU to GPU version, make all process free of CPU
    void assign (matrix *other)
    {
        printf ("Error for calling function: assign!\n");
    }

    void hostassign (matrix *other)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        memcpy (host_data, other->gethostdataptr(), Sizeof());
    }

    // assign the submatrix ([i0, j0] to [i0+nr-1, j0+nc-1]) using value from another matrx (other), both GPU and CPU
    void assignsubmatrix (matrix *other, int i0, int j0, int nr, int nc)
    {
        printf ("Error for calling function: assignsubmatrix!\n");
    }

    // initialize all element (both GPU and CPU) in matrx with v
    void initmatrix (int v = 0)
    {
        memset (host_data, v, Sizeof());
    }

    void assignmatvalue (real v)
    {
        printf ("Error for calling function: assignmatvalue!\n");
    }
    // sigmoid on all elements seperately
    void sigmoid ()
    {
        printf ("Error for calling function: sigmoid!\n");
    }

    void tanh ()
    {
        printf ("Error for calling function: tanh!\n");
    }

    void sigmoid_forchunk (int chunkiter, int chunksize)
    {
        printf ("Error for calling function: sigmoid_forchunk!\n");
    }

    void relu (float ratio)
    {
        printf ("Error for calling function: relu!\n");
    }

    void relu_forchunk (float ratio, int chunkiter, int chunksize)
    {
        printf ("Error for calling function: relu_forchunk!\n");
    }

    void gendropoutmask (float dropoutrate)
    {
        printf ("Error for calling function: gendropoutmask!\n");
    }

    void genEmbeddropoutmask (float dropoutrate)
    {
        printf ("Error for calling function: genEmbeddropoutmask!\n");
    }

    void genvardropoutmask (int mbidx, float dropoutrate)
    {
        printf ("Error for calling function: genvardropoutmask!\n");
    }

    void dropout (matrix *dropoutmask, float dropoutrate, bool evalmode)
    {
        printf ("Error for calling function: dropout!\n");
    }

    void hostrelu (float ratio)
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            if (host_data[i] > 0)
            {
                host_data[i] *= ratio;
            }
            else
            {
                host_data[i] = 0;
            }
        }
    }

    void hostsigmoid()
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            host_data[i] = 1/(1 + exp(-host_data[i]));
        }
    }

    void hosttanh ()
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            host_data[i] = (exp(2*host_data[i])-1)/(exp(2*host_data[i])+1);
        }
    }


    void hostsoftmax()
    {
        int a, maxi;
        float v, norm, maxv = 1e-8;
        assert (ncols == 1);
        maxv = 1e-10;
        for (a=0; a<nrows; a++)
        {
            v = host_data[a];
            if (v > maxv)
            {
                maxv = v;
                maxi = a;
            }
        }
        norm = 0;
        for (a=0; a<nrows; a++)
        {
            v = host_data[a] - maxv;
            host_data[a] = exp(v);
            norm += host_data[a];
        }
        for (a=0; a<nrows; a++)
        {
            v = host_data[a] / norm;
            host_data[a] = v;
        }
    }

    float hostcalentropy ()
    {
        float entropy = 0.0f;
        float prob;
        for (int i=0; i<nrows; i++)
        {
            prob = host_data[i];
            if (prob < 1e-10)
            {
                prob = 1e-10;
            }
            entropy += log(prob)*prob;
        }
        return -entropy;
    }

    void hostpartsoftmax(int swordid, int ewordid)
    {
        int a, maxi;
        float v, norm, maxv = 1e-8;
        assert (ncols == 1);
        maxv = 1e-10;
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a];
            if (v > maxv)
            {
                maxv = v;
                maxi = a;
            }
        }
        norm = 0;
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a] - maxv;
            host_data[a] = exp(v);
            norm += host_data[a];
        }
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a] / norm;
            host_data[a] = v;
        }
    }

    void softmax (matrix *lognorms)
    {
        printf ("Error for calling function: softmax!\n");
    }


    void gradcutoff (real gradient_cutoff)
    {
        printf ("Error for calling function: gradcutoff!\n");
    }

    // calculate error signal on the output layer (with softmax function)
    // er = (\delta(i,j) - ac), where i is the target, j is the index of nodes in output layer.
    // ac: ac
    // start_array: starts id for each sample (on GPU)
    // cn_array:    number of words need to calculate error in each sample (on GPU)
    void calerronoutputlayer(matrix *ac, int *words)
    {
        printf ("Error for calling function: calerronoutputlayer!\n");
    }

    void calerronoutputlayer_vr(matrix *ac, int *words, matrix *lognorms, float vrpenalty)
    {
        printf ("Error for calling function: calerronoutputlayer_vr!\n");
    }

    void hostadddotMultiply (matrix *c, matrix *p)
    {
        float *host_c = c->gethostdataptr ();
        float *host_p = p->gethostdataptr ();
        foreach_coord (i, j, c)
        {
            host_data[i+j*nrows] += host_c[i+j*nrows] * host_p[i+j*nrows];
        }
    }

    void hostadd (matrix *other)
    {
        if (rows() != other->rows() || cols() != other->cols())
        {
            printf ("error: hostadd: the size is different!\n");
            exit (0);
        }
        float *host_other = other->gethostdataptr();
        foreach_coord (i, j, other)
        {
            host_data[i+j*nrows] += host_other[i+j*nrows];
        }
    }

    void hostdotMultiply (matrix *other)
    {
        if (rows() != other->rows() || cols() != other->cols())
        {
            printf ("error: hostdotMultiply: the size is different!\n");
            exit (0);
        }
        float *host_other = other->gethostdataptr();
        foreach_coord (i, j, other)
        {
            host_data[i+j*nrows] *= host_other[i+j*nrows];
        }
    }

    void hostcalHiddenacGRU (matrix *h, matrix *z)
    {
        if (rows() != h->rows() || cols() != h->cols() || rows() != z->rows() || cols() != z->cols())
        {
            printf ("error: hostcalHiddenacGRU: the size is different!\n");
            exit (0);
        }
        float *host_h_ = h->gethostdataptr();
        float *host_z = z->gethostdataptr();
        foreach_coord (i, j, h)
        {
            float zvalue = host_z[i+j*nrows];
            float h_value = host_h_[i+j*nrows];
            float hvalue = host_data[i+j*nrows];
            host_data[i+j*nrows] = hvalue*(1-zvalue) + h_value*zvalue;
        }
    }

    void dotMultiply (matrix *other)
    {
        printf ("Error for calling function: dotMultiply!\n");
    }

    void calHiddenacGRU (matrix *x, matrix *h, matrix *z)
    {
        printf ("Error for calling function: calHiddenacGRU!\n");
    }

    void multiplyScalar (float v)
    {
        printf ("Error for calling function: multiplyScalar!\n");
    }

    void addScalar (float v)
    {
        printf ("Error for calling function: addScalar!\n");
    }

    // used when calculating the gradient in back propogation (through time)
    // er = er * ac * (1-ac)
    void multiplysigmoid (matrix *ac)
    {
        printf ("Error for calling function: multiplysigmoid!\n");
    }

    void multiplytanh (matrix *ac)
    {
        printf ("Error for calling function: multiplytanh!\n");
    }

    void multiplyrelue (matrix *ac, float ratio)
    {
        printf ("Error for calling function: multiplyrelue!\n");
    }
    // update the weight matrix connecting input word and hidden layer
    void updatelayer0_word (matrix *neu1_er, int *words, real alpha, real beta = 0.0)
    {
        printf ("Error for calling function: updatelayer0_word!\n");
    }


    // add with other matrix element by element
    // this += other
    void add (matrix *other)
    {
        printf ("Error for calling function: add!\n");
    }

    void addProduct (matrix *other1, matrix *other2)
    {
        printf ("Error for calling function: addProduct!\n");
    }

    void  adddotMultiply (matrix *peelhole, matrix *c)
    {
        printf ("Error for calling function: adddotMultiply!\n");
    }

    void addOneMinus (matrix *other)
    {
        printf ("Error for calling function: addOneMinus!\n");
    }

    void subtract (matrix *other)
    {
        printf ("Error for calling function: subtract!\n");
    }

    void addadagrad (matrix *dU, matrix *accsdU, float alpha, float l2reg)
    {
        printf ("Error for calling function: addadagrad!\n");
    }

    void addgrad (matrix *other, float alpha, float l2reg)
    {
        printf ("Error for calling function: addgrad!\n");
    }

    void addsquaregrad (matrix *other, float gamma, float beta)
    {
        printf ("Error for calling function: addsquaregrad!\n");
    }

    void addpeepholegrad (matrix *other, float alpha, float l2reg)
    {
        printf ("Error for calling function: addpeepholegrad!\n");
    }

    void addgrad_word (matrix *gradlayer0_word, int *prevwords, int minibatch)
    {
        printf ("Error for calling function: addgrad_word!\n");
    }

    // this = alpha * other + beta * this
    void addsubmatrix (matrix *other, int i0, int j0, int nr, int nc, real alpha=1.0, real beta = 1.0)
    {
        printf ("Error for calling function: addsubmatrix!\n");
    }

    void random(float min, float max)
    {
        int i, j;
        float v;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                v = randomv(min, max) + randomv(min,max) + randomv(min, max);
                host_data[i+j*nrows] = v;
            }
        }
    }

    void randomidentity (float scale)
    {
        printf ("Error for calling function: randomidentity!\n");
    }

    void sample (int *dev_samples, float *dev_randv, int minibatch)
    {
        printf ("Error for calling function: sample!\n");
    }

    void forwardWordlayer (matrix *srcac, matrix *tgtac, int *curclass, int *classinfo)
    {
        printf ("Error for calling function: forwardWordlayer!\n");
    }

    void softmaxWordlayer (int *curclass, int *classinfo)
    {
        printf ("Error for calling function: softmaxWordlayer!\n");
    }
    void calerronWordlayer (matrix *ac, int *curclass, int *curwords, int *classinfo)
    {
        printf ("Error for calling function: calerronWordlayer!\n");
    }

    void copyOutputWgtsforNCE (matrix *outputlayer, int *dev_targetsample, int ntargetsample, int *dev_ncesample, int nncesample)
    {
        printf ("Error for calling function: copyOutputWgtsforNCE!\n");
    }

    void calerronOutputLayer (matrix *neuN_ac_NCE, float *log_noise, int *curwords, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample)
    {
        printf ("Error for calling function: calerronOutputLayer!\n");
    }

    void calerronOutputLayer_oldversion (matrix *neuN_ac_NCE, matrix *neuN_er_NCE_mask, float *log_noise, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample)
    {
        printf ("Error for calling function: calerronOutputLayer_oldversion!\n");
    }

    void calnorm2 (int minibatch, float *norm2)
    {
        printf ("Error for calling function: calnorm2!\n");
    }

    void mulScalar (float gradient_cutoff)
    {
        printf ("Error for calling function: mulScalar!\n");
    }

    void L2norm (float gradient_cutoff, float *devnorm, int minibatch)
    {
        printf ("Error for calling function: L2norm!\n");
    }

    void addgrad_NCE (matrix *gradwgt, int *targetsample, int ntargetsample, int *ncesample, int nncesample, float alpha)
    {
        printf ("Error for calling function: addgrad_NCE!\n");
    }

    void Read (FILE *fptr)
    {
        int i, j;
        float v;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                fscanf (fptr, "%f ", &v);
                assignhostvalue(i, j, v);
            }
            fscanf (fptr, "\n");
        }
    }

    void Write (FILE *fptr)
    {
        int i, j;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                fprintf (fptr, "%.8f ", fetchhostvalue(i, j));
            }
            fprintf (fptr, "\n");
        }
    }

    void dump ()
    {
        int i, j;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                printf ("%.4f ", fetchhostvalue(i, j));
            }
            printf ("\n");
        }
        printf ("\n");
    }

};



void cumatrixXmatrix (matrix *A, matrix *B, matrix *C, bool transA, bool transB, real alpha = 1.0, real beta = 0.0, int Cbias = 0);

// tgter = alpha*layers*srcer + beta*tgter
void bperWordlayer (matrix *layers, matrix *srcer, matrix *tgter, int *curclass, int *classinfo, float alpha, float beta);
void bpupdateWordlayer (matrix *ac, matrix *er, matrix *layers, int *curclass, int *classinfo, float alpha, float beta);

#endif
