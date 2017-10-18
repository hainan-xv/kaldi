#ifndef _LAYER_H__
#define _LAYER_H__
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper.h"
#include "head.h"
#include "cudamatrix.h"
#include <algorithm>


class layer
{
public:
    size_t nrows;
    size_t ncols;
    size_t size;
    string type;
    matrix *U;
    int chunkiter, chunksize;
    int minibatch;      // used for L2norm
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac)
    {
        printf ("virtual function (host_forward) called!\n");
    }
    virtual void Read (FILE *fptr)
    {
        printf ("virtual function (Read) called!\n");
    }
    virtual void host_getWordEmbedding (int prevword, matrix *neu_ac)
    {
        printf ("virtual function (host_getWordEmbedding) called!\n");
    }
    virtual void host_resetHiddenac ()
    {
        printf ("virtual function (host_resetHiddenac) called!\n");
    }
    virtual void setTrainCrit (int n)
    {
        printf ("virtual function (setTrainCrit) called!\n");
    }
    virtual void setnodetype (int n)
    {
        printf ("virtual function (setnodetype) called!\n");
    }
    virtual void setreluratio (float v)
    {
        printf ("virtual function (setreluratio) called!\n");
    }
    virtual void initHiddenAc ()
    {
        printf ("virtual function (initHiddenAc) called!\n");
    }
    virtual void setLognormConst (float v)
    {
        printf ("virtual function (setLognormConst) called!\n");
    }
    virtual void host_copyLSTMhighwayc (layer *layer0)
    {
        printf ("virtual function (host_copyLSTMhighwayc) called!\n");
    }
    virtual void ReadFeaFile (string str)
    {
        printf ("virtual function (ReadFeaFile) called!\n");
    }
    virtual void host_assignFeaVec (int feaid)
    {
        printf ("virtual function (host_assignFeaVec) called!\n");
    }
    virtual int getnumfea ()
    {
        printf ("virtual function (getnumfea) called!\n");
    }
    virtual int getdimfea ()
    {
        printf ("virtual function (getdimfea) called!\n");
    }
    virtual matrix* gethidden_ac ()
    {
        printf ("virtual function (gethidden_ac) called!\n");
    }
    virtual matrix* gethidden_c ()
    {
        printf ("virtual function (gethidden_c) called!\n");
    }
    virtual void setCurword (int n)
    {
        printf ("virtual function (setCurword) called!\n");
    }
    virtual int GetHiddenSize ()
    {
        return 0;
    }
    void setChunkIter (int iter)  { chunkiter = iter;}
    layer (int nr, int nc, int mbsize, int cksize);
    ~layer ();
};

class inputlayer : public layer
{
private:
    ////// variable for topic feature input
    int dim_fea, num_fea;
    string feafile;
    matrix *feamatrix;
    matrix *U_fea;
    matrix *ac_fea;
    vector<matrix *> ac_fea_vec;
    int *mbfeaindices, *feaindices;
    //////

public:
    virtual void Read (FILE *fptr);

    inputlayer (int nr, int nc, int mbsize, int cksize, int dim_fea);
    ~inputlayer ();

    virtual void ReadFeaFile (string str);
    virtual void host_assignFeaVec (int feaid);
    virtual void host_getWordEmbedding (int prevword, matrix *neu_ac);
    virtual int getnumfea () { return num_fea; }
    virtual int getdimfea () { return dim_fea; }
};


class outputlayer : public layer
{
private:
    int  traincrit;         // CE, VR or NCE
    // for NCE training
    vector<matrix *> lognormvec;
    int *host_curwords;
    int outOOSindex;
    // for VR and NCE training
    float lognorm;
    int curword;
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void Read (FILE *fptr);

    outputlayer (int nr, int nc, int minibatch, int chunksize);
    ~outputlayer ();
    // NCE training

    virtual void setTrainCrit (int n)   { traincrit = n; }
    virtual void setLognormConst (float v) {lognorm = v;}
    virtual void setCurword (int n)     {curword = n;}
    };


class recurrentlayer : public layer
{
private:
    matrix *W, *hidden_ac;
    int  nodetype;          // sigmoid or relu
    float reluratio;
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_resetHiddenac ();
    virtual void Read (FILE *fptr);

    recurrentlayer (int nr, int nc, int minibatch, int chunksize);
    ~recurrentlayer ();

    virtual void setnodetype (int n) { nodetype = n; }
    virtual void setreluratio (float v) { reluratio = v; }
    virtual void initHiddenAc ();
    virtual matrix* gethidden_ac ()
    {
        return hidden_ac;
    }
    virtual int GetHiddenSize ()
    {
        return ncols;
    }
};

class feedforwardlayer : public layer
{
private:
    int  nodetype;          // sigmoid or relu
    float reluratio;
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void Read (FILE *fptr);

    feedforwardlayer (int nr, int nc, int mbsize, int cksize);

    virtual void setnodetype (int n) { nodetype = n; }
    virtual void setreluratio (float v)  { reluratio = v; }
};

class linearlayer : public layer
{
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void Read (FILE *fptr);

    linearlayer (int nr, int nc, int mbsize, int cksize);
};

class grulayer : public layer
{
protected:
    matrix *hidden_ac;
    matrix *Wr, *Wz, *Wh, *Ur, *Uz, *Uh;
    matrix *r, *z, *c, *h_;
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_resetHiddenac ();
    virtual void initHiddenAc ();
    virtual void Read (FILE *fptr);
    ~grulayer ();
    grulayer (int nr, int nc, int minibatch, int chunksize);
    virtual matrix* gethidden_ac ()
    {
        return hidden_ac;
    }
    virtual int GetHiddenSize ()
    {
        return ncols;
    }
};

class gruhighwaylayer : public grulayer
{
private:
    matrix *g, *v;
    matrix *Uhw, *Whw;
    vector <matrix *> g_vec, v_vec;
public:
    gruhighwaylayer (int nr, int nc, int minibatch, int chunksize);
    ~gruhighwaylayer ();
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);

    // virtual void host_resetHiddenac ();
    // virtual void initHiddenAc ();
    virtual void Read (FILE *fptr);
};

class lstmlayer : public layer
{
public:
    matrix *newc;
    vector<matrix *> c_vec;
protected:
    matrix *hidden_ac, *c_last;
    matrix *Uz, *Ui, *Uf, *Uo, *Wz, *Wi, *Wf, *Wo;
    matrix *Pi, *Pf, *Po;
    matrix *z, *i, *f, *c, *zi, *fc, *o;
public:
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_resetHiddenac ();
    virtual void initHiddenAc ();
    virtual void Read (FILE *fptr);
    virtual matrix* gethidden_ac ()
    {
        return hidden_ac;
    }
    virtual matrix* gethidden_c ()
    {
        return c;
    }
    virtual int GetHiddenSize ()
    {
        return ncols*2;
    }
    lstmlayer (int nr, int nc, int minibatch, int chunksize);
    ~lstmlayer ();
};


class lstmhighwaylayer : public lstmlayer
{
public:
    matrix *c_hw;
private:
    matrix *s, *sc;
    matrix *Uhw, *Phw, *Rhw;
    vector<matrix *> c_hw_vec, s_vec;
public:
    lstmhighwaylayer (int nr, int nc, int minibatch, int chunksize);
    ~lstmhighwaylayer ();
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);

    // virtual void host_resetHiddenac ();
    // virtual void initHiddenAc ();
    virtual void Read (FILE *fptr);
    virtual void host_copyLSTMhighwayc (layer *layer0);
};

#endif
