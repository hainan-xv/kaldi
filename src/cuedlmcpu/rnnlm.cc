#include "rnnlm.h"
#include <sstream>

#ifndef linux
#define exp10(x)	exp((x)*M_LN10)
#endif

bool isRNN (string type)
{
    if (type == "recurrent" || type == "gru" || type == "lstm" || type == "gru-highway" || type == "lstm-highway")
        return true;
    else
        return false;
}
bool isGRU (string type)
{
    if (type == "gru" || type == "gru-highway")
        return true;
    else
        return false;
}
bool isLSTM (string type)
{
    if (type == "lstm" || type == "lstm-highway")
        return true;
    else
        return false;

}
bool isLSTMhighway (string type)
{
    if (type == "lstm-highway")
        return true;
    else
        return false;
}

namespace cuedrnnlm {

void RNNLM::init()
{
    gradient_cutoff = 5;            // default no gradient cutoff
    llogp           = -1e8;
    min_improvement = 1.001;
    lognormconst    = 0;
    lognorm_mean    = 0;
    lognorm_var     = 0;
    lognorm_output  = -100.0;
    lrtunemode      = 0;                 // default use newbob
    alpha_divide    = false;
    flag_nceunigram = false;
    alpha           = 0.8;
    lambda          = 0.5;
    version         = 1.0;
    iter            = 0;
    num_layer       = 0;
    wordcn          = 0;
    trainwordcnt    = 0;
    validwordcnt    = 0;
    counter         = 0;
    num_oosword     = 0;
    dropoutrate     = 0;
    diaginit        = 0;
    host_prevwords  = NULL;
    host_curwords   = NULL;
    mbfeaindices    = NULL;
	nthread		    = 0;
    lmscale         = 12.0;
    ip              = 0;        // insertion penalty
}

RNNLM::~RNNLM()
{
    int i, j;
    if (host_prevwords)     free(host_prevwords);
    if (host_curwords)      free(host_curwords);
    for (i=0; i<num_layer; i++)
    {
        delete layers[i];
    }

    for (i=1; i<num_layer; i++)
    {
        for (j=0; j<chunksize; j++)
        {
            delete neu_ac_chunk[i][j];
        }
        delete neu_er[i];
    }
    if (mbfeaindices) {free(mbfeaindices); mbfeaindices=NULL;}
}


RNNLM::RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1, int fvocsize, bool bformat, int debuglevel, int mbsize/*=1*/, int cksize/*=1*/, int dev/*=0*/):inmodelfile(inmodelfile_1), inputwlist(inputwlist_1), outputwlist(outputwlist_1), binformat(bformat),  debug(debuglevel)
{
    int i;
    init ();
    minibatch = mbsize;
    chunksize = cksize;
    deviceid = dev;
    LoadRNNLM (inmodelfile);
    // this will be useful for sampling
    allocWordMem();
    setFullVocsize (fvocsize);
}

// allocate memory for RNNLM model
void RNNLM::allocRNNMem (bool flag_alloclayers/*=true*/)
{
    int i, j, nr, nc, dim_fea=0;
    num_layer = layersizes.size() - 1;
    if (num_layer < 2)
    {
        printf ("ERROR: the number of layers (%d) should be greater than 2\n", num_layer);
    }
    inputlayersize = layersizes[0];
    outputlayersize = layersizes[num_layer];
    layers.resize(num_layer);
    neu_ac.resize(num_layer+1);
    neu_er.resize(num_layer+1);
    if (flag_alloclayers)
    {
        for (i=0; i<num_layer; i++)
        {
            nr = layersizes[i];
            nc = layersizes[i+1];
            if (layertypes[i][0] == 'I' || layertypes[i][0] == 'i') // input layer
            {
                assert (i == 0);
                layers[i] = new inputlayer (nr, nc,  minibatch, chunksize, dim_fea);
                layertypes[i] = "input";
            }
            else if (layertypes[i][0] == 'L' || layertypes[i][0] == 'l') // linear layer
            {
                layers[i] = new linearlayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "linear";
            }
            else if (layertypes[i][0] == 'F' || layertypes[i][0] == 'f') // feedforward layer
            {
                layers[i] = new feedforwardlayer (nr, nc, minibatch, chunksize);
                if (layertypes[i].length() == 1 || layertypes[i][1] == '0')
                    // sigmoid node
                {
                    layers[i]->setnodetype (0);
                }
                else if (layertypes[i][1] == '1') // relu
                {
                    layers[i]->setnodetype (1);
                }
                layertypes[i] = "feedforward";
            }
            else if (layertypes[i][0] == 'r') // simple rnn with sigmoid
            {
                layers[i] = new recurrentlayer (nr, nc, minibatch, chunksize);
                layers[i]->setnodetype (0);
                layertypes[i] = "recurrent";
            }
            else if (layertypes[i][0] == 'R')       // simple rnn with relu
            {
                layers[i] = new recurrentlayer (nr, nc, minibatch, chunksize);
                layers[i]->setnodetype (1);
                layers[i]->setreluratio (RELURATIO);
                layertypes[i] = "recurrent";
            }
            else if (layertypes[i][0] == 'G' || layertypes[i][0] == 'g') // GRU
            {
                layers[i] = new grulayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "gru";
            }
            else if (layertypes[i][0] == 'X' || layertypes[i][0] == 'x') // GRU-highway
            {
                layers[i] = new gruhighwaylayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "gru-highway";
            }
            else if (layertypes[i][0] == 'M' || layertypes[i][0] == 'm') // LSTM
            {
                layers[i] = new lstmlayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "lstm";
            }
            else if (layertypes[i][0] == 'Y' || layertypes[i][0] == 'y') // LSTM-highway
            {
                // last layer much be lstm with same width
                layers[i] = new lstmhighwaylayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "lstm-highway";
            }
            else if (i == num_layer-1)
            {
                layers[i] = new outputlayer (nr, nc, minibatch, chunksize);
                layers[i]->setTrainCrit (traincritmode);
                layertypes[i] = "output";
            }
        }
    }

    neu_ac_chunk.resize (layersizes.size());
    for (i=1; i<layersizes.size(); i++)  // no need to allocate for input layer
    {
        int nrow = layersizes[i];
        neu_ac_chunk[i].resize(chunksize);
        for (j=0; j<chunksize; j++)
        {
            neu_ac_chunk[i][j] = new matrix (nrow, minibatch);
        }
        neu_er[i] = new matrix (nrow, minibatch);
        neu_ac[i] = new matrix (nrow, minibatch);
    }
}

// allocate memory used for input words, target words
void RNNLM::allocWordMem ()
{
    host_prevwords = (int *)calloc(minibatch*chunksize, sizeof(int));
    host_curwords = (int *)calloc (minibatch*chunksize, sizeof(int));
    memset(host_prevwords, 0, sizeof(int)*minibatch*chunksize);
    memset(host_curwords, 0, sizeof(int)*minibatch*chunksize);
}

void RNNLM::printPPLInfo ()
{
    string str;
    printf ("model file :       %s\n", inmodelfile.c_str());
    printf ("input  list:       %s\n", inputwlist.c_str());
    printf ("output list:       %s\n", outputwlist.c_str());
    printf ("num   layer:       %d\n", num_layer);
    for (int i=0; i<=num_layer; i++)
    {
        printf ("#layer[%d]  :       %-10d type:   %-6s\n", i, layersizes[i], layertypes[i].c_str());
    }
    printf ("independent:       %d\n", independent);
    printf ("test file  :       %s\n", testfile.c_str());
    printf ("nglm file  :       %s\n", nglmstfile.c_str());
    printf ("lambda (rnn):      %f\n", lambda);
    printf ("fullvocsize:       %d\n", fullvocsize);
    printf ("debug level:       %d\n", debug);
    printf ("nthread    :       %d\n", nthread);
    printf ("lognormconst :     %f\n", lognormconst);
}


void RNNLM::initHiddenAc ()
{
    for (int l=0; l<num_layer; l++)
    {
        if (isRNN(layertypes[l]))
        {
            layers[l]->initHiddenAc ();
        }
    }
}

void RNNLM::host_HandleSentEnd_fw ()
{
    if (! independent)
    {
        return;
    }
    for (int l=0; l<num_layer; l++)
    {
        if (isRNN (layertypes[l]))
        {
            layers[l]->host_resetHiddenac();
        }
    }
}

void RNNLM::ResetRechist ()
{
	host_HandleSentEnd_fw ();
}




void RNNLM::InitVariables ()
{
    int i, j;
    counter = 0;
    logp = 0;
    wordcn = 0;
}

void RNNLM::LoadRNNLM(string modelname)
{
    LoadTextRNNLM_new (modelname);
    ReadWordlist (inputwlist, outputwlist);
}


void RNNLM::LoadTextRNNLM_new (string modelname)
{
    int i, a, b, dim_fea;
    float v;
    char word[1024];
    FILE *fptr = NULL;
    // read model file
    fptr = fopen (modelname.c_str(), "r");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelname.c_str());
        exit (0);
    }
    fscanf (fptr, "cuedrnnlm v%f\n", &v);
    if (v != version)
    {
        printf ("Error: the version of rnnlm model(v%.1f) is not consistent with binary supported(v%.1f)\n", v, version);
        exit (0);
    }
    fscanf (fptr, "train file: %s\n", word);     trainfile = word;
    fscanf (fptr, "valid file: %s\n", word);     validfile = word;
    fscanf (fptr, "number of iteration: %d\n", &iter);
    iter ++;
    fscanf (fptr, "#train words: %d\n", &trainwordcnt);
    fscanf (fptr, "#valid words: %d\n", &validwordcnt);
    fscanf (fptr, "#layer: %d\n", &num_layer);
    layersizes.resize(num_layer+1);
    layertypes.resize(num_layer+1);
    for (i=0; i<layersizes.size(); i++)
    {
        fscanf (fptr, "layer %d size: %d type: %s\n", &b, &a, word);
        assert(b==i);
        layersizes[i] = a;
        layertypes[i] = word;
    }
    fscanf (fptr, "fullvoc size: %d\n", &fullvocsize);
    allocRNNMem (false);

    fscanf (fptr, "independent mode: %d\n", &independent);
    fscanf (fptr, "train crit mode: %d\n",  &traincritmode);
    fscanf (fptr, "log norm: %f\n", &lognorm_output);
    fscanf (fptr, "dim feature: %d\n", &dim_fea);
    lognormconst = lognorm_output;
    for (i=0; i<num_layer; i++)
    {
        fscanf (fptr, "layer %d -> %d type: %s\n", &a, &b, word);
        string type = layertypes[i];
        assert (word == type);
        int nr = layersizes[i];
        int nc = layersizes[i+1];
        if (type == "input")
        {
            assert (i == 0);
            layers[i] = new inputlayer (nr, nc, minibatch, chunksize, dim_fea);
        }
        else if (type == "output")
        {
            assert (i == num_layer-1);
            layers[i] = new outputlayer (nr, nc, minibatch, chunksize);
			layers[i]->setLognormConst (lognormconst);
        }
        else if (type == "feedforward")
        {
            layers[i] = new feedforwardlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "recurrent")
        {
            layers[i] = new recurrentlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "linear")
        {
            layers[i] = new linearlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "lstm")
        {
            layers[i] = new lstmlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "gru")
        {
            layers[i] = new grulayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "gru-highway")
        {
            layers[i] = new gruhighwaylayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "lstm-highway")
        {
            layers[i] = new lstmhighwaylayer (nr, nc, minibatch, chunksize);
        }
        else
        {
            printf ("Error: unknown layer type: %s\n", type.c_str());
        }
        layers[i]->Read(fptr);
    }
    fscanf (fptr, "%d", &a);
    if (a != CHECKNUM)
    {
        printf ("ERROR: failed to read the check number(%d) when reading model\n", CHECKNUM);
        exit (0);
    }
    if (debug > 1)
    {
        printf ("Successfully loaded model: %s\n", modelname.c_str());
    }
    fclose (fptr);
}

// read intput and output word list
void RNNLM::ReadWordlist (string inputlist, string outputlist)
{
    //index 0 for <s> and </s> in input and output layer
    //last node for <OOS>
    int i, a, b;
    float v;
    char word[1024];
    FILE *finlst, *foutlst;
    finlst = fopen (inputlist.c_str(), "r");
    foutlst = fopen (outputlist.c_str(), "r");
    if (finlst == NULL || foutlst == NULL)
    {
        printf ("ERROR: Failed to open input (%s) or output list file(%s)\n", inputlist.c_str(), outputlist.c_str());
        exit (0);
    }
    inputmap.insert(make_pair(string("<s>"), 0));
    outputmap.insert(make_pair(string("</s>"), 0));
    inputvec.clear();
    outputvec.clear();
    inputvec.push_back("<s>");
    outputvec.push_back("</s>");
    int index = 1;
    while (!feof(finlst))
    {
        if(fscanf (finlst, "%d%s", &i, word) == 2)
        {
            if (inputmap.find(word) == inputmap.end())
            {
                inputmap[word] = index;
                inputvec.push_back(word);
                index ++;
            }
        }
    }
    if (inputmap.find("<OOS>") == inputmap.end())
    {
        inputmap.insert(make_pair(string("<OOS>"), index));
        inputvec.push_back("<OOS>");
    }
    else
    {
        assert (inputmap["<OOS>"] == inputvec.size()-1);
    }

    index = 1;
    int clsid, prevclsid = 0;
    while (!feof(foutlst))
    {
        if (fscanf(foutlst, "%d%s", &i, word) == 2)
        {
            if (outputmap.find(word) == outputmap.end())
            {
                outputmap[word] = index;
                outputvec.push_back(word);
                index ++;
            }
        }
    }
    if (outputmap.find("<OOS>") == outputmap.end())
    {
        outputmap.insert(make_pair(string("<OOS>"), index));
        outputvec.push_back("<OOS>");
    }
    else
    {
        assert (outputmap["<OOS>"] == outputvec.size()-1);
    }
    assert (inputvec.size() == layersizes[0]);
    assert (outputvec.size() == layersizes[num_layer]);
    inStartindex = 0;
    outEndindex  = 0;
    inOOSindex   = inputvec.size() - 1;
    outOOSindex  = outputvec.size() - 1;
    assert (outOOSindex == outputmap["<OOS>"]);
    assert (inOOSindex == inputmap["<OOS>"]);
    fclose (finlst);
    fclose (foutlst);
}

void RNNLM::setIndependentmode (int v)
{
    independent = v;
}

void RNNLM::setFullVocsize (int n)
{
    if (n == 0)
    {
        fullvocsize = layersizes[0];
    }
    else
    {
        fullvocsize = n;
    }
}

bool RNNLM::calppl (string testfilename, float intpltwght, string nglmfile)
{
    int i, j, wordcn, nwordoov, cnt;
    vector<string> linevec;
    FILEPTR fileptr;
    float prob_rnn, prob_ng, prob_int, logp_rnn,
          logp_ng, logp_int, ppl_rnn, ppl_ng,
          ppl_int, entropy;
    bool flag_intplt = false, flag_oov = false;
    FILE *fptr=NULL, *fptr_nglm=NULL;
    auto_timer timer;
    timer.start();
    string word;
    testfile = testfilename;
    nglmstfile = nglmfile;
    lambda = intpltwght;
    if (debug > 1)
    {
        printPPLInfo ();
    }

#ifdef PRINTWENTROPY
    FILE *fptr_entropy = fopen ("./entropy.txt", "wt");
#endif
    if (!nglmfile.empty())
    {
        fptr_nglm = fopen (nglmfile.c_str(), "r");
        if (fptr_nglm == NULL)
        {
            printf ("ERROR: Failed to open ng stream file: %s\n", nglmfile.c_str());
            exit (0);
        }
        flag_intplt = true;
    }
    fileptr.open(testfile);

    wordcn = 0;
    nwordoov = 0;
    logp_int = 0;
    logp_rnn = 0;
    logp_ng = 0;
    entropy = 0;
    if (debug > 1)
    {
        if (flag_intplt)
        {
            printf ("\nId\tP_rnn\t\tP_ng\t\tP_int\t\tWord\n");
        }
        else
        {
            printf ("\nId\tP_rnn\t\tWord\n");
        }
    }
    while (!fileptr.eof())
    {
        if (layers[0]->getdimfea() > 0)
        {
            int feaid = fileptr.readint ();
            assert (feaid >= 0 && feaid < layers[0]->getnumfea());
            layers[0]->host_assignFeaVec (feaid);
        }
        fileptr.readline(linevec, cnt);
        if (linevec.size() > 0)
        {
            if (linevec[cnt-1] != "</s>")
            {
                linevec.push_back("</s>");
                cnt ++;
            }
            assert (cnt == linevec.size());
            if (linevec[0] == "<s>")    i = 1;
            else                        i = 0;
            prevword = inStartindex;
            host_HandleSentEnd_fw ();

            for (i; i<cnt; i++)
            {
                word = linevec[i];
                if (outputmap.find(word) == outputmap.end())
                {
                    curword = outOOSindex;
                }
                else
                {
                    curword = outputmap[word];
                }
                prob_rnn = host_forward (prevword, curword);

#ifdef PRINTWENTROPY
                float v = neu_ac[num_layer]->hostcalentropy ();
                entropy += v;
#endif

                if (curword == outOOSindex)     prob_rnn /= (fullvocsize-layersizes[num_layer]+1);

                if (flag_intplt)
                {
                    if (fscanf (fptr_nglm, "%f\n", &prob_ng) != 1)
                    {
                        printf ("ERROR: Failed to read ngram prob from ng stream file!\n");
                        exit (0);
                    }
                    if (fabs(prob_ng) < 1e-9)   {flag_oov = true;}
                    else                        flag_oov = false;
                }
                prob_int = lambda*prob_rnn + (1-lambda)*prob_ng;
                if (inputmap.find(word) == inputmap.end())
                {
                    prevword = inOOSindex;
                }
                else
                {
                    prevword = inputmap[word];
                }
                if (!flag_oov)
                {
                    logp_rnn += log10(prob_rnn);
                    logp_ng  += log10(prob_ng);
                    logp_int += log10(prob_int);
                }
                else
                {
                    nwordoov ++;
                }
                wordcn ++;
                if (debug > 1)
                {
                    if (flag_intplt)
                        printf ("%d\t%.10f\t%.10f\t%.10f\t%s", curword, prob_rnn, prob_ng, prob_int, word.c_str());
                    else
                        printf ("%d\t%.10f\t%s", curword, prob_rnn, word.c_str());
                    if (curword == outOOSindex)
                    {
                        if (flag_oov)   printf ("<OOV>");
                        else            printf ("<OOS>");
                    }
                    printf ("\n");
                }
                if (debug > 2)
                {
                    if (wordcn % 10000 == 0)
                    {
                        float nwordspersec = wordcn / (timer.stop());
                        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
                    }
                }
            }
#ifdef PRINTWENTROPY
                fprintf (fptr_entropy, "%.8f\n", entropy/cnt);
                entropy = 0;
#endif
        }
    }
    if (debug > 2)
    {
        float nwordspersec = wordcn / (timer.stop());
        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
    }
    ppl_rnn = exp10(-logp_rnn/(wordcn-nwordoov));
    ppl_ng  = exp10(-logp_ng/(wordcn-nwordoov));
    ppl_int = exp10(-logp_int/(wordcn-nwordoov));
    if (flag_intplt)
    {
        printf ("Total word: %d\tOOV word: %d\n", wordcn, nwordoov);
        printf ("N-Gram log probability: %.3f\n", logp_ng);
        printf ("RNNLM  log probability: %.3f\n", logp_rnn);
        printf ("Intplt log probability: %.3f\n\n", logp_int);
        printf ("N-Gram PPL : %.3f\n", ppl_ng);
        printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
        printf ("Intplt PPL : %.3f\n", ppl_int);
    }
    else
    {
        printf ("Total word: %d\tOOV word: %d\n", wordcn, nwordoov);
        printf ("Average logp: %f\n", logp_rnn/log10(2)/wordcn);
        printf ("RNNLM  log probability: %.3f\n", logp_rnn);
        printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
    }
    printf ("Avg Entropy: %.3f\n", entropy/wordcn);
#ifdef PRINTWENTROPY
    fclose (fptr_entropy);
#endif
    fileptr.close();

    if (fptr_nglm)
    {
        fclose(fptr_nglm);
    }
    return SUCCESS;
}

bool RNNLM::calnbest (string testfilename, float intpltwght, string nglmfile)
{
    int i, j, wordcn, cnt, nbestid, prevnbestid=-1, sentcnt=0, nword;
    vector<string> linevec, maxlinevec;
    FILEPTR fileptr;
    float prob_rnn, prob_ng, prob_int, logp_rnn,
          logp_ng, logp_int, ppl_rnn, ppl_ng,
          ppl_int, sentlogp, acscore, lmscore, score, maxscore;
    bool flag_intplt = false;
    FILE *fptr=NULL, *fptr_nglm=NULL;
    auto_timer timer;
    timer.start();
    string word;
    testfile = testfilename;
    nglmstfile = nglmfile;
    lambda = intpltwght;
    if (debug > 1)
    {
        printPPLInfo ();
    }
    if (!nglmfile.empty())
    {
        fptr_nglm = fopen (nglmfile.c_str(), "r");
        if (fptr_nglm == NULL)
        {
            printf ("ERROR: Failed to open ng stream file: %s\n", nglmfile.c_str());
            exit (0);
        }
        flag_intplt = true;
    }
    fileptr.open(testfile);

    wordcn = 0;
    logp_int = 0;
    logp_rnn = 0;
    logp_ng = 0;
    while (!fileptr.eof())
    {
        if (layers[0]->getdimfea() > 0)
        {
            int feaid = fileptr.readint ();
            assert (feaid >= 0 && feaid < layers[0]->getnumfea());
            layers[0]->host_assignFeaVec (feaid);
        }
        fileptr.readline(linevec, cnt);
        if (linevec.size() > 0)
        {
            if (linevec[cnt-1] != "</s>")
            {
                linevec.push_back("</s>");
                cnt ++;
            }
            assert (cnt == linevec.size());
            // the first two iterms for linevec are: <s> nbestid
            // nbid acscore  lmscore  nword <s> ... </s>
            // 0    2750.14 -6.03843    2   <s> HERE YEAH </s>
            // erase the first <s> and last</s>
            vector<string>::iterator it= linevec.begin();
            linevec.erase(it);
            cnt --;
            // it = linevec.end();
            // it --;
            // linevec.erase(it);
            nbestid = string2int(linevec[0]);

            if (nbestid != prevnbestid)
            {
                if (prevnbestid != -1)
                {
                    for (i=4; i<maxlinevec.size(); i++)
                    {
                        word = maxlinevec[i];
                        if (word != "<s>" && word != "</s>")
                        {
                            printf (" %s", word.c_str());
                        }
                    }
                    printf ("\n");
                }
                maxscore = -1e10;
            }

            acscore = string2float(linevec[1]);
            lmscore = string2float(linevec[2]);
            nword   = string2int(linevec[3]);
            if (linevec[4] == "<s>")    i = 5;
            else                        i = 4;
            sentlogp = 0;
            prevword = inStartindex;
            host_HandleSentEnd_fw ();
            for (i; i<cnt; i++)
            {
                word = linevec[i];
                if (outputmap.find(word) == outputmap.end())
                {
                    curword = outOOSindex;
                }
                else
                {
                    curword = outputmap[word];
                }
                prob_rnn = host_forward (prevword, curword);
                if (curword == outOOSindex)     prob_rnn /= (fullvocsize-layersizes[num_layer]+1);

                if (flag_intplt)
                {
                    if (fscanf (fptr_nglm, "%f\n", &prob_ng) != 1)
                    {
                        printf ("ERROR: Failed to read ngram prob from ng stream file!\n");
                        exit (0);
                    }
                }
#ifdef LOGLINEARINT
                prob_int = pow(prob_rnn, lambda)*pow(prob_ng, 1-lambda);
#else
                prob_int = lambda*prob_rnn + (1-lambda)*prob_ng;
#endif

                if (inputmap.find(word) == inputmap.end())
                {
                    prevword = inOOSindex;
                }
                else
                {
                    prevword = inputmap[word];
                }
                logp_rnn += log10(prob_rnn);
                logp_ng  += log10(prob_ng);
                logp_int += log10(prob_int);
                sentlogp += log10(prob_int);
                wordcn ++;
                if (debug == 1)
                {
                    printf ("%f ", log10(prob_int));
                }
                if (debug > 1)
                {
                    if (flag_intplt)
                        printf ("%d\t%.10f\t%.10f\t%.10f\t%s", curword, prob_rnn, prob_ng, prob_int, word.c_str());
                    else
                        printf ("%d\t%.10f\t%s", curword, prob_rnn, word.c_str());
                    if (curword == outOOSindex)
                    {
                        printf ("<OOS>");
                    }
                    printf ("\n");
                }
                if (debug > 1)
                {
                    if (wordcn % 10000 == 0)
                    {
                        float nwordspersec = wordcn / (timer.stop());
                        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
                    }
                }
            }
            sentcnt ++;
            if (debug == 1)
            {
                printf ("sent=%f %d\n", sentlogp, sentcnt);
            }
#if 0
            if (debug == 0)
            {
                printf ("%f\n", sentlogp);
            }
#endif
            if (debug == 0)
            {
                score = acscore + sentlogp*lmscale + ip*(nword+1);
                if (score > maxscore)
                {
                    maxscore = score;
                    maxlinevec = linevec;
                }
                for (i=4; i<cnt; i++)
                {
                    word = linevec[i];
                    // printf (" %s", word.c_str());
                }
                // printf ("\n");
            }
            fflush(stdout);
            prevnbestid = nbestid;
        }
    }
    for (i=4; i<maxlinevec.size(); i++)
    {
        word = maxlinevec[i];
        if (word != "<s>" && word != "</s>")
        {
            printf (" %s", word.c_str());
        }
    }
    printf ("\n");
    if (debug > 1)
    {
        float nwordspersec = wordcn / (timer.stop());
        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
    }
    ppl_rnn = exp10(-logp_rnn/(wordcn));
    ppl_ng  = exp10(-logp_ng/(wordcn));
    ppl_int = exp10(-logp_int/(wordcn));
    if (debug > 1)
    {
        if (flag_intplt)
        {
            printf ("Total word: %d\n", wordcn);
            printf ("N-Gram log probability: %.3f\n", logp_ng);
            printf ("RNNLM  log probability: %.3f\n", logp_rnn);
            printf ("Intplt log probability: %.3f\n\n", logp_int);
            printf ("N-Gram PPL : %.3f\n", ppl_ng);
            printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
            printf ("Intplt PPL : %.3f\n", ppl_int);
        }
        else
        {
            printf ("Total word: %d\n", wordcn);
            printf ("Average logp: %f\n", logp_rnn/log10(2)/wordcn);
            printf ("RNNLM  log probability: %.3f\n", logp_rnn);
            printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
        }
    }
    fileptr.close();

    if (fptr_nglm)
    {
        fclose(fptr_nglm);
    }
    return SUCCESS;
}

float RNNLM::host_forward (int prevword, int curword)
{
   int i;
   layers[0]->host_getWordEmbedding (prevword, neu_ac[1]);
   for (i=1; i<num_layer; i++)
   {
	   if (i == num_layer-1)
	   {
		   layers[i]->setCurword(curword);
	   }
       layers[i]->host_forward (neu_ac[i], neu_ac[i+1]);
       // if next layer is LSTM-highway layer
       if (i != num_layer-1 && isLSTMhighway (layertypes[i+1]))
       {
           layers[i+1]->host_copyLSTMhighwayc (layers[i]);
       }
   }
   float ans = neu_ac[num_layer]->fetchhostvalue(curword, 0);
   assert(ans == ans);
   return ans;
}

// added for Kaldi
int RNNLM::getOneHiddenLayerSize()
{
  return layersizes[num_layer-1];
}

void RNNLM::setFullVocabSize(int fvocsize)
{
  fullvocsize = fvocsize;
}

void RNNLM::saveContextToVector(std::vector <float> *context_out)
{
  assert(context_out != NULL);
  context_out->resize(layersizes[num_layer-1]);

  float *srcac;
#if 0
  srcac = neu_ac[num_layer]->gethostdataptr();
  for (int i = 0; i  <  layersizes[num_layer-1]; i++)
  {
    (*context_out)[i] = srcac[i];
  }
#else
  // printf ("xiexie0: size=%d\n", (*context_out).size());
  srcac = neu_ac[num_layer-1]->gethostdataptr();
  for (int i = 0; i  <  layersizes[num_layer-1]; i++)
  {
    (*context_out)[i] = srcac[i];
  }
#endif
}

void RNNLM::restoreContextFromVector(const std::vector<float> &context_in)
{
  float *dstac;
  dstac = layers[num_layer-2]->gethidden_ac()->gethostdataptr();

  assert(context_in.size() == layersizes[num_layer-1]);
  for (int i = 0; i  <  layersizes[num_layer-1]; i++) {
    dstac[i] = context_in[i];
  }
}

bool RNNLM::calLogProb(string input, int fvocsize, float &output_logp, float &output_ppl)
{
  int i, j, wordcn, cnt;
  vector<string> linevec;
  float prob_rnn, logp_rnn, ppl_rnn;
  string word;

  fullvocsize = fvocsize;
  wordcn = 0;
  logp_rnn = 0;

  stringstream ssin(input);
  while(std::getline(ssin, word, ' '))
  {
    linevec.push_back(word);
  }
  cnt = linevec.size();
  if(cnt>0)
  {
    if(linevec[cnt-1] != "</s>")
    {
      linevec.push_back("</s>");
      cnt++;
    }
    if(linevec[0] == "<s>") i=1;
    else i=0;
    prevword = inStartindex;    // index for <s>
    host_HandleSentEnd_fw ();
    for(; i<cnt; i++)
    {
      word = linevec[i];
      if(outputmap.find(word) == outputmap.end())
      {
        curword = outOOSindex;
      }
      else
      {
        curword = outputmap[word];
      }
      prob_rnn = host_forward (prevword, curword);
      if (curword == outOOSindex)
      {
        j = (fullvocsize - layersizes[num_layer]);
        if(j>0)
        prob_rnn /= (float) j;
      }
      if (inputmap.find(word) == inputmap.end())
      {
        prevword = inOOSindex;
      }
      else
      {
        prevword = inputmap[word];
      }
      logp_rnn += log(prob_rnn);
      wordcn++;
    }
    ppl_rnn = exp10(-logp_rnn/log(10)/wordcn);
  }
  else
  {
    return false;
  }

  output_logp = logp_rnn;
  output_ppl = ppl_rnn;

  return true;
}



float RNNLM::computeConditionalLogprob(std::string current_word, const std::vector<std::string> &history_words, const std::vector<float> &context_in, std::vector<float> *context_out)
{
//  cout << "current word is " << current_word << endl;
  int i, j, cnt;
  vector<string> linevec;
  float prob_rnn, logp_rnn;
  string word;

  restoreContextFromVector(context_in);
  for (int n = 0; n < 10; n++) {
    cout << context_in[n] << ", ";
  }
  cout << endl;
  logp_rnn = 0;

#if 1
  printf ("xie0: curword=%s ", current_word.c_str());
  for (i=0; i<history_words.size(); i++)
  {
	  printf ("hist[%d]=%s ", i, history_words[i].c_str());
  }
  printf ("\n");
#endif


  for(i=0; i<history_words.size(); i++)
  {
    linevec.push_back(history_words[i]);
  }
  cnt = linevec.size();
  if(cnt==0)
  {
    linevec.push_back("<s>");
    cnt++;
  }

  word = linevec[cnt-1];
//  cout << "pre: " << word << endl;
  if(inputmap.find(word) == inputmap.end())
  {
    prevword = outOOSindex;
  }
  else
  {
    prevword = outputmap[word];
  }
  word = current_word;
  if(outputmap.find(word) == outputmap.end())
  {
    curword = outOOSindex;
  }
  else
  {
    curword = outputmap[word];
  }
  prob_rnn = host_forward (prevword, curword);

  cout << prob_rnn << endl;
//  if (prob_rnn != prob_rnn) {cout << curword << " " << outOOSindex << " " << current_word << endl; prob_rnn = 1e-20;}

  if (curword == outOOSindex)
  {
    cout << "this is an OOS" << endl;
    j = (fullvocsize - layersizes[num_layer]);
    cout << "j is " << fullvocsize << " - " << layersizes[num_layer] << " ";
    cout << "before: " << prob_rnn;
    if(j>0)
    prob_rnn /= (float) j;
    cout << " after: " << prob_rnn << endl;
  }

  logp_rnn += log(prob_rnn);

  if (context_out != NULL)
  {
    saveContextToVector(context_out);
  }

#if 1
  if (context_out != NULL && context_in.size() > 0 && (*context_out).size() > 0)
  {
    printf ("xiexie1 context_in[0]=%f,  context_out[0]=%f,  prob_rnn=%f\n", context_in[0], (*context_out)[0], prob_rnn);
  }
#endif

#if 0
  if (context_in.size() > 0 && (*context_out).size() > 0)
  {
	  printf ("xiexie1 context_in[0]=%f, context_out[0]=%f, prob_rnn=%f\n", context_in[0], (*context_out)[0], prob_rnn);
  }
#endif

  return logp_rnn;
}

void RNNLM::loadfresh(const string &inmodelfile_1, const string &inputwlist_1, const string &outputwlist_1, bool bformat, int mbsize, int debuglevel)
{
    inmodelfile = inmodelfile_1;
    inputwlist = inputwlist_1;
    outputwlist = outputwlist_1;
    binformat = bformat;
    minibatch = mbsize;
    debug = debuglevel;

    init ();
    minibatch = 1;
    LoadRNNLM (inmodelfile);

    if (fullvocsize == 0)
    {
        fullvocsize = layersizes[0];
    }
}

RNNLM::RNNLM(vector<int> &lsizes, int fvocsize):layersizes(lsizes), fullvocsize(fvocsize)
{
}



}
