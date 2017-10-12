#include "head.h"
#include "Mathops.h"

class FILEPTR
{
protected:
    FILE *fptr;
    int i;
    string filename;
public:
    FILEPTR()
    {
        fptr = NULL;
    }
    ~FILEPTR()
    {
        if (fptr)       fclose(fptr);
        fptr = NULL;
    }
    void open (string fn)
    {
        filename = fn;
        fptr = fopen (filename.c_str(), "rt");
        if (fptr == NULL)
        {
            printf ("ERROR: Failed to open file: %s\n", filename.c_str());
            exit (0);
        }
    }
    void close()
    {
        if (fptr)
        {
            fclose(fptr);
            fptr = NULL;
        }
    }
    bool eof()
    {
        return feof(fptr);
    }
    int readint ()
    {
        if (!feof(fptr))
        {
            if(fscanf (fptr, "%d", &i) != 1)
            {
                if (!feof(fptr))
                {
                    printf ("Warning: failed to read feature index from text file (%s)\n", filename.c_str());
                    exit (0);
                }
            }
            return i;
        }
        else
        {
            return INVALID_INT;
        }
    }
    void readline (vector<string> &linevec, int &cnt)
    {
        linevec.clear();
        char word[1024];
        char c;
        int index=0;
        cnt = 0;
        while (!feof(fptr))
        {
            c = fgetc(fptr);
            // getvalidchar (fptr, c);
            if (c == '\n')
            {
                if (cnt==0 && word[0] != '<')
                {
                    linevec.push_back("<s>");
                    cnt ++;
                }
                if (index > 0)
                {
                    word[index] = 0;
                    linevec.push_back(word);
                    cnt ++;
                }
                break;
            }
            else if ((c == ' ' || c == '\t') && index == 0) // space in the front of line
            {
                continue;
            }
            else if ((c == ' ' || c=='\t') && index > 0) // space in the middle of line
            {
                word[index] = 0;
                if (cnt==0 && word[0] != '<')
                {
                    linevec.push_back("<s>");
                    cnt ++;
                }
                linevec.push_back(word);
                index = 0;
                cnt ++;
            }
            else
            {
                word[index] = c;
                index ++;
            }
        }
        if (cnt>0 && word[0] != '<')
        {
            linevec.push_back("</s>");
            cnt ++;
        }
    }
};

class ReadFileBuf
{
protected:
    int linecnt, wordcnt, cachesize, minibatch, lineperstream,
        mbcnt, mbcnter, randint;
    char line[1024][100];
    string filename, inputfilename, outputfilename, inputindexfilename, outputindexfilename;
    FILEPTR fileptr;
    FILE *fptr_in, *fptr_out;
    int *Indata, *Outdata, *featureindices;
    double *unigram, *accprob;
    float *logunigram;
    Matrix *inputbufptr, *outputbufptr;
    WORDMAP &inputmap, &outputmap;
public:
    ReadFileBuf(string txtfile, WORDMAP &inmap, WORDMAP &outmap, int mbsize, int csize, int num_fea=0, int rint=-1);
    void Indexword(string word, int &Inindex, int &Outindex);
    void FillBuffer();
    void DeleteIndexfile();
    void Init();
    void GetData(int index, int *indata, int *outdata)
    {
        Matrix &inputbuf = *inputbufptr;
        Matrix &outputbuf = *outputbufptr;
        memcpy (indata, inputbuf[index], sizeof(int)*minibatch);
        memcpy (outdata, outputbuf[index], sizeof(int)*minibatch);
    }
    ~ReadFileBuf()
    {
        if (inputbufptr)       delete inputbufptr;
        if (outputbufptr)      delete outputbufptr;
        if (Indata)         delete Indata;
        if (Outdata)        delete Outdata;
        if (unigram)        delete [] unigram;
        if (accprob)        delete [] accprob;
        if (logunigram)     delete [] logunigram;
        if (featureindices) delete [] featureindices;
        if (fptr_in)        fclose(fptr_in);
        if (fptr_out)       fclose(fptr_out);
    }

    int getWordcnt ()
    {
        return wordcnt;
    }
    int getLinecnt ()
    {
        return linecnt;
    }
    int getMBcnt ()
    {
        return mbcnt;
    }
    int getRandint()
    {
        return randint;
    }
    void setMBcnter (int n)
    {
        mbcnter = n;
    }
    double* getUnigram ()
    {
        return unigram;
    }
    double* getAccprob ()
    {
        return accprob;
    }
    float* getLogUnigram ()
    {
        return logunigram;
    }
    int* getfeaptr()
    {
        return featureindices;
    }
};
