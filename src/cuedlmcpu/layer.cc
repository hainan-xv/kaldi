#include "layer.h"

void DeleteMat (matrix *ptr)
{
    if (ptr)
    {
        delete ptr;
        ptr = NULL;
    }
}
void DeleteMatVec (vector<matrix *> matvec)
{
    for (int i=0; i<matvec.size(); i++)
    {
        if (matvec[i])
        {
            delete matvec[i];
            matvec[i] = NULL;
        }
    }
}

void matrixXvector (float *src, float *wgt, float *dst, int nr, int nc)
{
    int i, j;
#ifdef NUM_THREAD
#pragma omp parallel for num_threads(NUM_THREAD)
#endif
    for (i=0; i<nc; i++)
    {
        for (j=0; j<nr; j++)
        {
            dst[i] += src[j]*wgt[j+i*nr];
        }
    }
    return;
}

layer::~layer()
{
    DeleteMat (U);
}

layer::layer (int nr, int nc, int mbsize, int cksize)
{
    type = "base layer";
    nrows = nr;
    ncols = nc;
    size = nr*nc;
    U = new matrix (nrows, ncols);
    U->random (MINRANDINITVALUE, MAXRANDINITVALUE);

    chunksize = cksize;
    minibatch = mbsize;
}


void inputlayer::Read (FILE *fptr)
{
    U->Read (fptr);
    if (dim_fea > 0)
    {
        U_fea->Read(fptr);
    }
}

void outputlayer::Read (FILE *fptr)
{
    U->Read (fptr);
}

void recurrentlayer::Read (FILE *fptr)
{
    fscanf (fptr, "nodetype: %d\n", &nodetype);
    fscanf (fptr, "reluratio: %f\n", &reluratio);
    U->Read (fptr);
    W->Read (fptr);
}
void grulayer::Read (FILE *fptr)
{
    Wh->Read(fptr);
    Uh->Read(fptr);
    Wr->Read(fptr);
    Ur->Read(fptr);
    Wz->Read(fptr);
    Uz->Read(fptr);
}
void gruhighwaylayer::Read (FILE *fptr)
{
    Wh->Read(fptr);
    Uh->Read(fptr);
    Wr->Read(fptr);
    Ur->Read(fptr);
    Wz->Read(fptr);
    Uz->Read(fptr);
    // highway part
    Uhw->Read(fptr);
    Whw->Read(fptr);
}
void lstmlayer::Read (FILE *fptr)
{
    Wz->Read (fptr);
    Uz->Read (fptr);
    Wi->Read (fptr);
    Ui->Read (fptr);
    Wf->Read (fptr);
    Uf->Read (fptr);
    Wo->Read (fptr);
    Uo->Read (fptr);
    Pi->Read (fptr);
    Pf->Read (fptr);
    Po->Read (fptr);
}
void lstmhighwaylayer::Read (FILE *fptr)
{
    Wz->Read (fptr);
    Uz->Read (fptr);
    Wi->Read (fptr);
    Ui->Read (fptr);
    Wf->Read (fptr);
    Uf->Read (fptr);
    Wo->Read (fptr);
    Uo->Read (fptr);
    Pi->Read (fptr);
    Pf->Read (fptr);
    Po->Read (fptr);
    // highway part
    Uhw->Read(fptr);
    Phw->Read(fptr);
    Rhw->Read(fptr);
}

void feedforwardlayer::Read (FILE *fptr)
{
    fscanf (fptr, "nodetype: %d\n", &nodetype);
    fscanf (fptr, "reluratio: %f\n", &reluratio);
    U->Read (fptr);
}


void linearlayer::Read (FILE *fptr)
{
    U->Read (fptr);
}

void recurrentlayer::host_resetHiddenac ()
{
    for (int i=0; i<ncols; i++)
    {
        hidden_ac->assignhostvalue (i, 0, RESETVALUE);
    }
}
void grulayer::host_resetHiddenac ()
{
    for (int i=0; i<ncols; i++)
    {
        hidden_ac->assignhostvalue (i, 0, RESETVALUE);
    }
}
void lstmlayer::host_resetHiddenac ()
{
    for (int i=0; i<ncols; i++)
    {
        hidden_ac->assignhostvalue (i, 0, RESETVALUE);
        c->assignhostvalue (i, 0, RESETVALUE);
    }
}

void outputlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    memset (dstac, 0, ncols*sizeof(float));
	if (lognorm > 0)  // NCE or VR model
	{
		for (int i=0; i<nrows; i++)
		{
			dstac[curword] += srcac[i]*wgts[i+curword*nrows];
		}
		dstac[curword] = exp (dstac[curword] - lognorm);
	}
	else		// CE trained model
	{
		matrixXvector (srcac, wgts, dstac, nrows, ncols);
		neu1_ac->hostsoftmax ();
	}
}

void recurrentlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
    float *hiddensrcac = hidden_ac->gethostdataptr();
    float *recwgts = W->gethostdataptr();
    matrixXvector (hiddensrcac, recwgts, dstac, ncols, ncols);
    if (nodetype == 0) // sigmoid
    {
        neu1_ac->hostsigmoid();
    }
    else if (nodetype == 1) // relue
    {
        neu1_ac->hostrelu (reluratio);
    }
    hidden_ac->hostassign (neu1_ac);
}

void grulayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Ur = Ur->gethostdataptr();
    float *host_Wr = Wr->gethostdataptr();
    float *host_Uz = Uz->gethostdataptr();
    float *host_Wz = Wz->gethostdataptr();
    float *host_Uh = Uh->gethostdataptr();
    float *host_Wh = Wh->gethostdataptr();

    float *host_srcac = neu0_ac->gethostdataptr();
    float *host_dstac = neu1_ac->gethostdataptr();
    float *host_hiddenac=hidden_ac->gethostdataptr();
    float *host_r = r->gethostdataptr();
    float *host_z = z->gethostdataptr();
    float *host_h_ = h_->gethostdataptr();

    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, ncols, ncols);
    z->hostsigmoid();

    memset (host_r, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ur, host_r, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wr, host_r, ncols, ncols);
    r->hostsigmoid();
    r->hostdotMultiply (hidden_ac);

    memset (host_h_, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uh, host_h_, nrows, ncols);
    matrixXvector (host_r, host_Wh, host_h_, ncols, ncols);
    h_->hosttanh();

    memset (host_dstac, 0, ncols*sizeof(float));
    hidden_ac->hostcalHiddenacGRU (h_, z);
    neu1_ac->hostassign (hidden_ac);
}


void gruhighwaylayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Ur = Ur->gethostdataptr();
    float *host_Wr = Wr->gethostdataptr();
    float *host_Uz = Uz->gethostdataptr();
    float *host_Wz = Wz->gethostdataptr();
    float *host_Uh = Uh->gethostdataptr();
    float *host_Wh = Wh->gethostdataptr();

    float *host_srcac = neu0_ac->gethostdataptr();
    float *host_dstac = neu1_ac->gethostdataptr();
    float *host_hiddenac=hidden_ac->gethostdataptr();
    float *host_r = r->gethostdataptr();
    float *host_z = z->gethostdataptr();
    float *host_h_ = h_->gethostdataptr();

    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, ncols, ncols);
    z->hostsigmoid();

    memset (host_r, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ur, host_r, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wr, host_r, ncols, ncols);
    r->hostsigmoid();
    r->hostdotMultiply (hidden_ac);

    memset (host_h_, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uh, host_h_, nrows, ncols);
    matrixXvector (host_r, host_Wh, host_h_, ncols, ncols);
    h_->hosttanh();

    memset (host_dstac, 0, ncols*sizeof(float));
    v->hostassign (hidden_ac);
    v->hostcalHiddenacGRU (h_, z);

    // highway part
    float *host_g = g->gethostdataptr ();
    float *host_Uhw = Uhw->gethostdataptr ();
    float *host_Whw = Whw->gethostdataptr ();
    memset (host_g, 0, ncols*sizeof(float));

    matrixXvector (host_srcac, host_Uhw, host_g, nrows, ncols);
    matrixXvector (host_hiddenac, host_Whw, host_g, nrows, ncols);
    g->hostsigmoid();

    hidden_ac->hostassign (neu0_ac);
    hidden_ac->hostcalHiddenacGRU (v, g);
    neu1_ac->hostassign (hidden_ac);
}

void lstmlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Uz = Uz->gethostdataptr ();
    float *host_Wz = Wz->gethostdataptr ();
    float *host_Ui = Ui->gethostdataptr ();
    float *host_Wi = Wi->gethostdataptr ();
    float *host_Uf = Uf->gethostdataptr ();
    float *host_Wf = Wf->gethostdataptr ();
    float *host_Uo = Uo->gethostdataptr ();
    float *host_Wo = Wo->gethostdataptr ();
    float *host_srcac = neu0_ac->gethostdataptr ();
    float *host_dstac = neu1_ac->gethostdataptr ();
    float *host_hiddenac = hidden_ac->gethostdataptr ();
    float *host_z = z->gethostdataptr ();
    float *host_i = i->gethostdataptr ();
    float *host_f = f->gethostdataptr ();
    float *host_o = o->gethostdataptr ();

    // compute input gate
    memset (host_i, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ui, host_i, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wi, host_i, ncols, ncols);
    i->hostadddotMultiply(c, Pi);
    i->hostsigmoid ();

    // compute forget gate
    memset (host_f, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uf, host_f, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wf, host_f, ncols, ncols);
    f->hostadddotMultiply (c, Pf);
    f->hostsigmoid ();
    f->hostdotMultiply (c);

    // compute block input
    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, ncols, ncols);
    z->hosttanh ();
    z->hostdotMultiply (i);

    // compute cell state
    newc->hostassign (z);
    newc->hostadd (f);

    c->hostassign (newc);

    // compute output gate
    memset (host_o, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uo, host_o, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wo, host_o, ncols, ncols);
    o->hostadddotMultiply (newc, Po);
    o->hostsigmoid();

    hidden_ac->hostassign (newc);
    hidden_ac->hosttanh ();
    hidden_ac->hostdotMultiply (o);
    neu1_ac->hostassign (hidden_ac);
}

void lstmhighwaylayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Uz = Uz->gethostdataptr ();
    float *host_Wz = Wz->gethostdataptr ();
    float *host_Ui = Ui->gethostdataptr ();
    float *host_Wi = Wi->gethostdataptr ();
    float *host_Uf = Uf->gethostdataptr ();
    float *host_Wf = Wf->gethostdataptr ();
    float *host_Uo = Uo->gethostdataptr ();
    float *host_Wo = Wo->gethostdataptr ();
    float *host_srcac = neu0_ac->gethostdataptr ();
    float *host_dstac = neu1_ac->gethostdataptr ();
    float *host_hiddenac = hidden_ac->gethostdataptr ();
    float *host_z = z->gethostdataptr ();
    float *host_i = i->gethostdataptr ();
    float *host_f = f->gethostdataptr ();
    float *host_o = o->gethostdataptr ();

    // compute input gate
    memset (host_i, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ui, host_i, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wi, host_i, nrows, ncols);
    i->hostadddotMultiply(c, Pi);
    i->hostsigmoid ();

    // compute forget gate
    memset (host_f, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uf, host_f, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wf, host_f, nrows, ncols);
    f->hostadddotMultiply (c, Pf);
    f->hostsigmoid ();
    f->hostdotMultiply (c);

    // compute block input
    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, nrows, ncols);
    z->hosttanh ();
    z->hostdotMultiply (i);

    // compute cell state
    newc->hostassign (z);
    newc->hostadd (f);

    /******* highway part *********/
    float *host_s = s->gethostdataptr ();
    float *host_Uhw = Uhw->gethostdataptr ();
    memset (host_s, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uhw, host_s, nrows, ncols);
    s->hostadddotMultiply (c, Phw);
    s->hostadddotMultiply (c_hw, Rhw);
    s->hostsigmoid ();
    s->hostdotMultiply (c_hw);
    newc->hostadd (s);
    /******************************/

    c->hostassign (newc);

    // compute output gate
    memset (host_o, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uo, host_o, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wo, host_o, nrows, ncols);
    o->hostadddotMultiply (newc, Po);
    o->hostsigmoid();

    hidden_ac->hostassign(newc);
    hidden_ac->hosttanh ();
    hidden_ac->hostdotMultiply (o);
    neu1_ac->hostassign (hidden_ac);
}

void feedforwardlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
    if (nodetype == 0) // sigmoid
    {
        neu1_ac->hostsigmoid();
    }
    else if (nodetype == 1) // relue
    {
        neu1_ac->hostrelu (reluratio);
    }
}

void linearlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
}

void lstmhighwaylayer::host_copyLSTMhighwayc (layer *layer0)
{
    lstmlayer *lstmlayer0 = dynamic_cast <lstmlayer *> (layer0);
    c_hw->hostassign (lstmlayer0->newc);
}

inputlayer::~inputlayer()
{
    DeleteMat (U_fea);
    DeleteMat (feamatrix);
    DeleteMatVec (ac_fea_vec);
}


inputlayer::inputlayer (int nr, int nc, int mbsize, int cksize, int dim): layer(nr, nc, mbsize, cksize)
{
    type = "input";
    dim_fea = dim;
    feamatrix = NULL;
    mbfeaindices = NULL;
    feaindices = NULL;
    num_fea = 0;
    if (dim_fea == 0)
    {
        U_fea = NULL;
        ac_fea = NULL;
        ac_fea_vec.clear();
    }
}

outputlayer::~outputlayer ()
{
    int i;
    DeleteMatVec (lognormvec);
}

outputlayer::outputlayer (int nr, int nc, int mbsize, int cksize): layer (nr, nc, mbsize, cksize)
{
    type = "output";
    traincrit = 0;          // CE training by default
    lognormvec.resize(chunksize);
    for (int i=0; i<chunksize; i++)
    {
        lognormvec[i] = new matrix (1, minibatch);
        lognormvec[i]->initmatrix ();
    }
}


recurrentlayer::~recurrentlayer ()
{
    int i;
    DeleteMat (W);
    DeleteMat (hidden_ac);
}
recurrentlayer::recurrentlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "recurrent";
    W = new matrix (ncols, ncols);
    W->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    hidden_ac = new matrix (ncols, minibatch);
    hidden_ac->initmatrix();
}

lstmlayer::~lstmlayer ()
{
    DeleteMat (newc);
    DeleteMatVec (c_vec);
    DeleteMat (hidden_ac);
    DeleteMat (Uz);
    DeleteMat (Ui);
    DeleteMat (Uf);
    DeleteMat (Uo);
    DeleteMat (Wz);
    DeleteMat (Wi);
    DeleteMat (Wf);
    DeleteMat (Wo);
    DeleteMat (Pi);
    DeleteMat (Pf);
    DeleteMat (Po);
    DeleteMat (z);
    DeleteMat (i);
    DeleteMat (f);
    DeleteMat (c);
    DeleteMat (zi);
    DeleteMat (fc);
    DeleteMat (o);
}
lstmlayer::lstmlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "lstm";

    delete U;
    U = NULL;
    Uz = new matrix (nrows, ncols);
    Ui = new matrix (nrows, ncols);
    Uf = new matrix (nrows, ncols);
    Uo = new matrix (nrows, ncols);

    Uz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Ui->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uf->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uo->random (MINRANDINITVALUE, MAXRANDINITVALUE);

    Wz = new matrix (ncols, ncols);
    Wi = new matrix (ncols, ncols);
    Wf = new matrix (ncols, ncols);
    Wo = new matrix (ncols, ncols);

    Wz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wi->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wf->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wo->random (MINRANDINITVALUE, MAXRANDINITVALUE);

    Pi = new matrix (ncols, 1);
    Pf = new matrix (ncols, 1);
    Po = new matrix (ncols, 1);
    Pi->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Pf->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Po->random (MINRANDINITVALUE, MAXRANDINITVALUE);

    newc = new matrix (ncols, minibatch);
    zi = new matrix (ncols, minibatch);
    fc = new matrix (ncols, minibatch);

    zi->initmatrix ();
    fc->initmatrix ();

    // when ppl and nbest, they need to use hidden_ac, c, etc.
    hidden_ac = new matrix (ncols, minibatch);
    c = new matrix (ncols, minibatch);
    z = new matrix (ncols, minibatch);
    i = new matrix (ncols, minibatch);
    f = new matrix (ncols, minibatch);
    o = new matrix (ncols, minibatch);

}

lstmhighwaylayer::~lstmhighwaylayer()
{
    DeleteMat (c_hw);
    DeleteMat (s);
    DeleteMat (sc);
    DeleteMat (Uhw);
    DeleteMat (Phw);
    DeleteMat (Rhw);
}

lstmhighwaylayer::lstmhighwaylayer (int nr, int nc, int mbsize, int cksize) : lstmlayer (nr, nc, mbsize, cksize)
{
    type = "lstm-highway";
    assert (nrows == ncols);        // for highway connection, the nrows=ncols
    Uhw = new matrix (nrows, ncols);
    Phw = new matrix (ncols, 1);
    Rhw = new matrix (nrows, 1);
    sc   = new matrix (nrows, minibatch);
    Uhw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Phw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Rhw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    sc->initmatrix ();

    // when ppl and nbest, they need to use hidden_ac, c, etc.
    s = new matrix (ncols, minibatch);
    s->initmatrix ();
}

grulayer::~grulayer ()
{
    DeleteMat (hidden_ac);
    DeleteMat (Wr);
    DeleteMat (Wz);
    DeleteMat (Wh);
    DeleteMat (Ur);
    DeleteMat (Uz);
    DeleteMat (Uh);
    DeleteMat (r);
    DeleteMat (z);
    DeleteMat (c);
    DeleteMat (h_);
}

grulayer::grulayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "gru";

    delete U;
    U = NULL;
    Uh = new matrix (nrows, ncols);
    Ur  = new matrix (nrows, ncols);
    Uz = new matrix (nrows, ncols);

    Wh = new matrix (ncols, ncols);
    Wr  = new matrix (ncols, ncols);
    Wz = new matrix (ncols, ncols);

    Wh->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wr->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uh->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Ur->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uz->random (MINRANDINITVALUE, MAXRANDINITVALUE);


    // when ppl and nbest, they need to use hidden_ac, c, etc.
    hidden_ac = new matrix (ncols, minibatch);
    r = new matrix (ncols, minibatch);
    z = new matrix (ncols, minibatch);
    h_ = new matrix (ncols, minibatch);
    c = new matrix (ncols, minibatch);
    hidden_ac->initmatrix ();
    r->initmatrix ();
    z->initmatrix ();
    c->initmatrix ();
    h_->initmatrix ();
}

gruhighwaylayer::~gruhighwaylayer()
{
    DeleteMat (g);
    DeleteMat (v);
    DeleteMat (Uhw);
    DeleteMat (Whw);
}
gruhighwaylayer::gruhighwaylayer (int nr, int nc, int mbsize, int cksize) : grulayer (nr, nc, mbsize, cksize)
{
    type = "gru-highway";
    assert (nrows == ncols);        // for highway connection, the nrows=ncols
    Uhw  = new matrix (nrows, ncols);
    Whw  = new matrix (ncols, ncols);


    Uhw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Whw->random (MINRANDINITVALUE, MAXRANDINITVALUE);


    // when ppl and nbest, they need to use hidden_ac, c, etc.
    g = new matrix (ncols, minibatch);
    v = new matrix (ncols, minibatch);
    g->initmatrix ();
    v->initmatrix ();
}

feedforwardlayer::feedforwardlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize) { type = "feedforward";}

linearlayer::linearlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize) {type = "linear";}


void inputlayer::ReadFeaFile (string filestr)
{
    feafile = filestr;
    int i, j, t;
    float value;
    FILE *fptr = fopen (feafile.c_str(), "r");
    if (fptr == NULL)
    {
        printf ("Error: Failed to open feature file: %s\n", feafile.c_str());
        exit(0);
    }
    fscanf (fptr, "%d %d", &num_fea, &dim_fea);
    // if the fea file is two large, just allocate cpu memory
    feamatrix = new matrix (dim_fea, num_fea);
    feamatrix->initmatrix();
    printf ("%d lines feature (with %d dimensions) will be read from %s\n", num_fea, dim_fea, feafile.c_str());
    i = 0;
    while (i < num_fea)
    {
        if (feof(fptr))         break;
        fscanf (fptr, "%d", &j);
        assert (j == i);
        for (t=0; t<dim_fea; t++)
        {
            fscanf (fptr, "%f", &value);
            feamatrix->assignhostvalue(t, i, value);
        }
        i ++;
    }
    if (i != num_fea)
    {
        printf ("Warning: only read %d lines from the feature file: %s, should be %d lines\n", i, feafile.c_str(), num_fea);
    }
    feamatrix->assign();
    printf ("%d feature lines (with %d dimensions) is read from %s successfully\n", num_fea, dim_fea,  feafile.c_str());
    fclose(fptr);
}



void inputlayer::host_assignFeaVec (int feaid)
{
    float *feaptr = feamatrix->gethostdataptr (0, feaid);
    float *acptr  = ac_fea->gethostdataptr ();
    memcpy (acptr, feaptr, sizeof(float)*dim_fea);
}

void inputlayer::host_getWordEmbedding (int prevword, matrix *neu_ac)
{
    float *wgts = U->gethostdataptr ();
    for (int i=0; i<ncols; i++)
    {
        neu_ac->assignhostvalue(i, 0, wgts[prevword+nrows*i]);
    }
}


void recurrentlayer::initHiddenAc ()
{
    hidden_ac->assignmatvalue (RESETVALUE);
}
void grulayer::initHiddenAc ()
{
    hidden_ac->assignmatvalue (RESETVALUE);
}
void lstmlayer::initHiddenAc ()
{
    hidden_ac->assignmatvalue (RESETVALUE);
    c->assignmatvalue (RESETVALUE);
}

