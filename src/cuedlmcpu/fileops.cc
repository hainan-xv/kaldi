#include "head.h"
#include "fileops.h"

float random(float min, float max)
{
#ifndef RAND48
    return rand()/(real)RAND_MAX*(max-min)+min;
#else
    return drand48()*(max-min)+min;
#endif
}

ReadFileBuf::ReadFileBuf(string txtfile, WORDMAP &inmap, WORDMAP &outmap, int mbsize, int csize, int num_fea/*= false*/, int rint/* = -1*/): filename(txtfile), inputmap(inmap), outputmap(outmap), minibatch(mbsize), cachesize(csize), Indata(NULL), Outdata(NULL), inputbufptr(NULL), outputbufptr(NULL), mbcnter(0)
{
    vector<string> linevec;
    char str[1024];
    string word;
    int mbindex, i, nvalid, cnt, Inindex, Outindex;
    bool flag_feature = false;
    featureindices = NULL;
    Indata = new int [minibatch];
    Outdata = new int [minibatch];
    // use additional input feature or not
    if (num_fea > 0) flag_feature = true;
    if (rint == -1)
    {
        randint = clock();
    }
    else
    {
        randint = rint;
    }
    sprintf (str, "%d", randint);
    string strrandint = str;
    fileptr.open(txtfile);
    linecnt = 0;
    wordcnt = 0;
    while (!fileptr.eof())
    {
        fileptr.readline(linevec, cnt);
        if (linevec.size() > 0)         // do not count empty line
        {
            linecnt ++;
        }
        wordcnt += cnt;
    }
    lineperstream = linecnt/minibatch + 1;
    fileptr.close();

    if (flag_feature)
    {
        featureindices = new int [linecnt+1];
    }

    FILE **finputptr;
    FILE **foutputptr;
    finputptr = new FILE*[minibatch];
    foutputptr = new FILE*[minibatch];
    for (i=0; i<minibatch; i++)
    {
        sprintf (str, "%d", i);
        string stri = str;
        inputfilename = txtfile + ".input.index.mb" + stri + "." + strrandint;
        outputfilename = txtfile + ".output.index.mb" + stri + "." + strrandint;
        finputptr[i] = fopen (inputfilename.c_str(), "wb");
        foutputptr[i] = fopen (outputfilename.c_str(), "wb");
        if (finputptr[i]==NULL || foutputptr[i]==NULL)
        {
            printf ("ERROR: Failed to write input (%s) or output (%s) index file!\n", inputfilename.c_str(), outputfilename.c_str());
            exit (0);
        }
    }

    vector<string> inputvec, outputvec;
    inputvec.resize(inputmap.size()+5);
    outputvec.resize(outputmap.size()+5);
    for (map<string, int>::iterator it=inputmap.begin(); it!=inputmap.end(); it++)
    {
        string word = it->first;
        int index = it->second;
        inputvec[index] = word;
    }
    for (map<string, int>::iterator it=outputmap.begin(); it!=outputmap.end(); it++)
    {
        string word = it->first;
        int index = it->second;
        outputvec[index] = word;
    }

    int outputlayersize = outputmap.size();
    unigram = new double [outputlayersize];
    accprob = new double [outputlayersize];
    logunigram = new float [outputlayersize];
    memset(unigram, 0, sizeof(double)*outputlayersize);
    long int totalcnt = 0;

    linecnt = 0;
    fileptr.open(txtfile);
    while (!fileptr.eof())
    {
        if (flag_feature)
        {
            featureindices[linecnt] = fileptr.readint();
            assert (featureindices[linecnt] < num_fea);
        }
        fileptr.readline(linevec, cnt);
        int cnt1 = 0, cnt2 = 0;
        if (linevec.size() > 0)     // skip empty line
        {
            // mbindex = linecnt / lineperstream;
            mbindex = linecnt % minibatch;
            linecnt ++;
            assert (mbindex < minibatch);
            for (int i=0; i<cnt; i++)
            {
                word = linevec[i];
                Indexword (word, Inindex, Outindex);
                if (Inindex >= 0)
                {
                    cnt1 ++;
                    fwrite (&Inindex, sizeof(int), 1, finputptr[mbindex]);
                }
                if (Outindex >= 0)
                {
                    cnt2++;
                    fwrite (&Outindex, sizeof(int), 1, foutputptr[mbindex]);
                    unigram[Outindex] += 1;
                    totalcnt ++;
                }
            }
            if (cnt1 != cnt2)
            {
                printf ("Check the text file, the number of input word (%d) and output word (%d) is not the same\n", cnt1, cnt2);
                printf ("The text line is: \n");
                for (int i=0; i<cnt; i++)
                {
                    word = linevec[i];
                    printf ("%d %s ", i, word.c_str());
                }
                printf ("\n");
                exit (0);
            }
        }
    }


    float eta = 1.0;
    float norm = 0;
    for (i=0; i<outputmap.size(); i++)
    {
        unigram[i] /= totalcnt;
        unigram[i] = powf(unigram[i], eta);
        norm += unigram[i];
    }

    // compute the unigram information
    for (i=0; i<outputmap.size(); i++)
    {
#ifdef NCE_NOISE_UNIGRAM
        unigram[i] = 1.0 / outputmap.size();
#else
        unigram[i] /= norm;
#endif
        if (i == 0)     accprob[i] = unigram[i];
        else            accprob[i] = accprob[i-1]+unigram[i];
        logunigram[i] = log(unigram[i]);
    }

    for (i=0; i<minibatch; i++)
    {
        fclose (finputptr[i]);
        fclose (foutputptr[i]);
    }
    // read index file
    for (i=0; i<minibatch; i++)
    {
        sprintf (str, "%d", i);
        string stri = str;
        inputfilename = txtfile+".input.index.mb"+stri+"."+strrandint;
        outputfilename = txtfile+".output.index.mb"+stri+"."+strrandint;
        finputptr[i] = fopen (inputfilename.c_str(), "rb");
        foutputptr[i] = fopen (outputfilename.c_str(), "rb");
        if (finputptr[i]==NULL || foutputptr[i]==NULL)
        {
            printf ("ERROR: Failed to open input (%s) or output (%s) index file!\n", inputfilename.c_str(), outputfilename.c_str());
            exit (0);
        }
    }
    inputfilename = txtfile + ".input.index." + strrandint;
    outputfilename = txtfile + ".output.index." + strrandint;
    fptr_in = fopen (inputfilename.c_str(), "wb");
    fptr_out = fopen (outputfilename.c_str(), "wb");
    if (fptr_in==NULL || fptr_out==NULL)
    {
        printf ("ERROR: Failed to create input (%s) or output (%s) index file!\n", inputfilename.c_str(), outputfilename.c_str());
        exit (0);
    }


    nvalid = minibatch;
    mbcnt = 0;
    while (nvalid > 0)
    {
        nvalid = 0;
        int nvalid_out = 0;
        for (i=0; i<minibatch; i++)
        {
            if (!feof(finputptr[i]))
            {
                if (fread (Indata+i, sizeof(int), 1, finputptr[i]) == 1)
                {
                    nvalid ++;
                }
                else
                {
                    Indata[i] = INVALID_INT;
                }
            }
            else
            {
                Indata[i] = INVALID_INT;
            }

            if (!feof(foutputptr[i]))
            {
                if (fread (Outdata+i, sizeof(int), 1, foutputptr[i]) == 1)
                {
                    nvalid_out ++;
                }
                else
                {
                    Outdata[i] = INVALID_INT;
                }
            }
            else
            {
                Outdata[i] = INVALID_INT;
            }
            if (nvalid != nvalid_out)
            {
                printf ("ERROR here, #in=%d, #out=%d!\n", nvalid, nvalid_out);
                for (int a=0; a<nvalid; a++)
                {
                    printf ("%d -> %d\n", Indata[a], Outdata[a]);
                }
                exit (0);
            }
        }
        if (nvalid > 0)
        {
            mbcnt ++;
            fwrite(Indata, sizeof(int), minibatch, fptr_in);
            fwrite(Outdata, sizeof(int), minibatch, fptr_out);
#if 0
            for (int i=0; i<minibatch; i++)
            {
                string inword, outword;
                if (Indata[i] == INVALID_INT)  inword = "NULL"; else inword = inputvec[Indata[i]];
                if (Outdata[i] == INVALID_INT) outword = "NULL"; else outword = outputvec[Outdata[i]];
                printf ("%5d\tin:%5d(%s)\tout:%5d(%s)\t", i, Indata[i], inword.c_str(), Outdata[i], outword.c_str());
            }
            printf ("\n");
#endif
        }
    }

    // clean mbindex file and close the file point
    for (i=0; i<minibatch; i++)
    {
        sprintf (str, "%d", i);
        string stri = str;
        inputfilename = txtfile + ".input.index.mb" + stri + "." + strrandint;
        outputfilename = txtfile + ".output.index.mb" + stri + "." + strrandint;
        if( remove(inputfilename.c_str()) )
        {
            printf ("ERROR: Failed to remove %s\n", inputfilename.c_str());
        }
        if( remove(outputfilename.c_str()) )
        {
            printf ("ERROR: Failed to remove %s\n", outputfilename.c_str());
        }
        fclose (finputptr[i]);
        fclose (foutputptr[i]);
    }
    fclose (fptr_in);
    fclose (fptr_out);

    delete [] finputptr;
    delete [] foutputptr;

    if (cachesize == 0)
    {
        inputbufptr = new Matrix (minibatch, mbcnt + 5);
        outputbufptr = new Matrix (minibatch, mbcnt + 5);
    }
    else
    {
        inputbufptr = new Matrix (minibatch, cachesize);
        outputbufptr = new Matrix (minibatch, cachesize);
    }
    inputindexfilename = filename + ".input.index." + strrandint;
    outputindexfilename = filename + ".output.index." + strrandint;
    fptr_in = fopen (inputindexfilename.c_str(), "rb");
    fptr_out = fopen (outputindexfilename.c_str(), "rb");
}

void ReadFileBuf::Init()
{
    mbcnter = 0;
    if (fptr_in)    fclose (fptr_in);
    if (fptr_out)   fclose (fptr_out);
    fptr_in = fopen (inputindexfilename.c_str(), "rb");
    fptr_out = fopen (outputindexfilename.c_str(), "rb");
}

void ReadFileBuf::Indexword(string word, int &Inindex, int &Outindex)
{
    int inStartindex = inputmap["<s>"];
    int outEndindex  = outputmap["</s>"];
    int inOOSindex   = inputmap["<OOS>"];
    int outOOSindex  = outputmap["<OOS>"];
    if (inputmap.find(word) == inputmap.end())
    {
        Inindex = inOOSindex;
    }
    else
    {
        Inindex = inputmap[word];
        if (Inindex == inStartindex)
        {
            Outindex = -1;
            return;
        }
    }

    if (outputmap.find(word) == outputmap.end())
    {
        Outindex = outOOSindex;
    }
    else
    {
        Outindex = outputmap[word];
        if (Outindex == outEndindex)
        {
            Inindex = -1;
        }
    }

}

void ReadFileBuf::FillBuffer()
{
    int i;
    Matrix &inputbuf = *inputbufptr;
    Matrix &outputbuf = *outputbufptr;
    if (cachesize > 0)
    {
        for (i=0; i<cachesize; i++)
        {
            mbcnter ++;
            if (mbcnter <= mbcnt)
            {
                fread (inputbuf[i], sizeof(int), minibatch, fptr_in);
                fread (outputbuf[i], sizeof(int), minibatch, fptr_out);
            }
        }
    }
    else
    {
        for (i=0; i<mbcnt; i++)
        {
            fread (inputbuf[i], sizeof(int), minibatch, fptr_in);
            fread (outputbuf[i], sizeof(int), minibatch, fptr_out);
        }
    }
}

void ReadFileBuf::DeleteIndexfile()
{
    if( remove(inputindexfilename.c_str()) )
    {
        printf ("ERROR: Failed to remove input index file: %s\n", inputindexfilename.c_str());
    }
    if( remove(outputindexfilename.c_str()) )
    {
        printf ("ERROR: Failed to remove output index file: %s\n", outputindexfilename.c_str());
    }
}
