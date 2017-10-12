#ifndef __HEAD_HELPER__
#define __HEAD_HELPER__
#include "head.h"

void printusage(char *str)
{
    printf ("Usage of command \"%s\"\n", str);
    printf ("Function:\n");
    printf ("%35s\t%s\n", "-train                   :",                    "RNNLM training (GPU supported)");
    printf ("%35s\t%s\n", "-ppl                     :",                      "RNNLM evaluation for perplexity (CPU supported)");
    printf ("%35s\t%s\n", "-nbest                   :",                    "RNNLM evaluation for N best rescoring (CPU supported)" );
    printf ("%35s\t%s\n", "-sample                  :",                   "sampling words from RNNLM (GPU supported)");
    printf ("Configuration:\n");
    printf ("%35s\t%s\n", "-trainfile   <string>    :",     "specify the text file for RNNLM training");
    printf ("%35s\t%s\n", "-validfile   <string>    :",     "specify the valid file for RNNLM training");
    printf ("%35s\t%s\n", "-testfile    <string>    :",     "specify the test file for RNNLM evaluation");
    printf ("%35s\t%s\n", "-feafile     <string>    :",     "specify the feature matrix file");
    printf ("%35s\t%s\n", "-device      <int>       :",        "specify the GPU id for RNNLM training (default: 0)");
    printf ("%35s\t%s\n", "-minibatch   <int>       :",        "specify the minibatch size for RNNLM training (default: 32)");
    printf ("%35s\t%s\n", "-chunksize   <int>       :",        "specify the chunk size for RNNLM training (default: 32)");

    printf ("%35s\t%s\n", "-layers <int:int:int:...>:", "specify the model structure of RNNLM (including input and output layer)");
    printf ("%35s\t%s\n", "-pretrainepoch <int>     :", "specify the number of epoch for pretraining when training RNNLM with more than 1 hidden layer");
    printf ("%35s\t%s\n", "-nodetype <int>          :", "specify the type of hidden node.[0 (sigmoid, default) | 1 (relu)]");
    printf ("%35s\t%s\n", "-reluratio <int>         :", "specify the ratio of relu when relu used as hidden node. (default: 0.5)");
    printf ("%35s\t%s\n", "-clipping <float>        :", "specify the clipping for bp error (default: 5)");
    printf ("%35s\t%s\n", "-traincrit   <string>    :",     "specify the training criterion for RNNLM [ce (default) | nce | vr]");
    printf ("%35s\t%s\n", "-lrtune      <string>    :",     "specify the method of learning rate tuning for RNNLM training [newbob (default) | adagrad | rmsprop]");
    printf ("%35s\t%s\n", "-inputwlist  <string>    :",     "specify the input word list for RNNLM training");
    printf ("%35s\t%s\n", "-outputwlist <string>    :",     "specify the output word list for RNNLM training");
    printf ("%35s\t%s\n", "-learnrate   <float>     :",      "specify the initial learning rate for RNNLM training (default: 0.8)");
    printf ("%35s\t%s\n", "-momentum   <float>      :",      "specify the momentum for RNNLM training (default: 0.0)");
    printf ("%35s\t%s\n", "-pretrainlearnrate <float>:",      "specify the initial learning rate for pretraining training of RNNLM with more than 1 hidden layer (default: 0.8)");
    printf ("%35s\t%s\n", "-vrpenalty   <float>     :",      "specify the penalty for RNNLM training with variance regularization (default: 0.0)");
    printf ("%35s\t%s\n", "-ncesample   <int>       :",        "specify the sample number for NCE based RNNLM training (default: 10)");
    printf ("%35s\t%s\n", "-nclass      <int>       :",        "specify the number of class in output layer (default: 0), if it is greater than 0, class based RNNLM will be trained");
    printf ("%35s\t%s\n", "-lognormconst <float>       :",        "specify the log norm const for NCE training and evaluation without normalization (default: -1.0)");
    printf ("%35s\t%s\n", "-lambda      <float>     :",      "specify the interpolation weight for RNNLM when interpolating with N-Gram LM (default: 0.5)");
    printf ("%35s\t%s\n", "-cachesize   <int>       :",        "specify the cache size for RNNLM training (default: 0)");
    printf ("%35s\t%s\n", "-debug       <int>       :",        "specify the debug level (default: 1)");
    printf ("%35s\t%s\n", "-nthread     <int>       :",        "specify the number of thread for computation (default: 1)");
    printf ("%35s\t%s\n", "-randseed    <int>       :",        "specify the rand seed to generate rand value (default: 1)");
    printf ("%35s\t%s\n", "-readmodel   <string>    :",     "specify the RNNLM model to be read");
    printf ("%35s\t%s\n", "-writemodel  <string>    :",     "specify the RNNLM model to be written");
    printf ("%35s\t%s\n", "-fullvocsize <int>       :",        "specify the full vocabulary size, all OOS words will share the probability");
    printf ("%35s\t%s\n", "-lmscale <float>         :",        "specify the lmscale, used for nbest rescoring");
    printf ("%35s\t%s\n", "-ip <float>              :",        "specify the insertion penalty, used for nbest rescoring");
    printf ("%35s\t%s\n", "-independent <int>       :",        "specify sentence independent or dependent mode (default: 1)");
    printf ("%35s\t%s\n", "-binformat               :",                "specify the model will be read or write with binary format (default: false)");
    printf ("%35s\t%s\n", "-min_improvement <float> :",  "specify the minimum improvement to stop RNNLM training (default: 1.003)");
    printf ("%35s\t%s\n", "-nglmstfile  <string>    :",     "specify the ngram lm stream file for interpolation");
    printf ("%35s\t%s\n", "-nsample     <int>       :",        "specify number of sample word from RNNLM (default: 1000)");
    printf ("%35s\t%s\n", "-unigramfile <string>    :",     "specify unigram lm file");
    printf ("%35s\t%s\n", "-sampletextfile <string> :",  "specify text file for sampling words from RNNLM");
    printf ("%35s\t%s\n", "-nceunigramfile <string> :",     "specify unigram lm file for NCE training");
    printf ("%35s\t%s\n", "-nceunigramintpltwgt <float>:",     "specify the interpolation weight between unigram on train data and external unigram lm in NCE training");
    printf ("\nexample:\n");
    printf ("%s -train -trainfile data/train.dat -validfile data/dev.dat -device 1 -minibatch 64 -chunksize 32 -layers 31858:200:20002 -traincrit ce -lrtune newbob -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist  -debug 2 -randseed 1 -writemodel h200.mb64/rnnlm.txt -independent 1 -learnrate 1.0  -min_improvement 1.003\n", str);
    printf ("%s -ppl -readmodel h200.mb64/rnnlm.txt -testfile data/test.dat -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist -nglmstfile ng.st -lambda 0.5 -debug 2\n", str);
    printf ("%s -nbest -readmodel h200.mb64/rnnlm.txt.nbest -testfile data/test.dat  -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist -nglmstfile ng.st -lambda 0.5 -debug 2\n", str);
}

void printevalusage(char *str)
{
    printf ("Usage of command \"%s\"\n", str);
    printf ("Function:\n");
    printf ("%35s\t%s\n", "-ppl                     :",                      "RNNLM evaluation for perplexity (CPU supported)");
    printf ("%35s\t%s\n", "-nbest                   :",                    "RNNLM evaluation for N best rescoring (CPU supported)" );
    printf ("%35s\t%s\n", "-sample                  :",                   "sampling words from RNNLM (GPU supported)");
    printf ("Configuration:\n");
    printf ("%35s\t%s\n", "-validfile   <string>    :",     "specify the valid file for RNNLM training");
    printf ("%35s\t%s\n", "-testfile    <string>    :",     "specify the test file for RNNLM evaluation");
    printf ("%35s\t%s\n", "-feafile     <string>    :",     "specify the feature matrix file");
    printf ("%35s\t%s\n", "-inputwlist  <string>    :",     "specify the input word list for RNNLM training");
    printf ("%35s\t%s\n", "-outputwlist <string>    :",     "specify the output word list for RNNLM training");
    printf ("%35s\t%s\n", "-lognormconst <float>       :",        "specify the log norm const for NCE training and evaluation without normalization (default: -1.0)");
    printf ("%35s\t%s\n", "-lambda      <float>     :",      "specify the interpolation weight for RNNLM when interpolating with N-Gram LM (default: 0.5)");
    printf ("%35s\t%s\n", "-debug       <int>       :",        "specify the debug level (default: 1)");
    printf ("%35s\t%s\n", "-nthread     <int>       :",        "specify the number of thread for computation (default: 1)");
    printf ("%35s\t%s\n", "-randseed    <int>       :",        "specify the rand seed to generate rand value (default: 1)");
    printf ("%35s\t%s\n", "-readmodel   <string>    :",     "specify the RNNLM model to be read");
    printf ("%35s\t%s\n", "-fullvocsize <int>       :",        "specify the full vocabulary size, all OOS words will share the probability");
    printf ("%35s\t%s\n", "-lmscale <float>         :",        "specify the lmscale, used for nbest rescoring");
    printf ("%35s\t%s\n", "-ip <float>              :",        "specify the insertion penalty, used for nbest rescoring");
    printf ("%35s\t%s\n", "-independent <int>       :",        "specify sentence independent or dependent mode (default: 1)");
    printf ("%35s\t%s\n", "-binformat               :",                "specify the model will be read or write with binary format (default: false)");
    printf ("%35s\t%s\n", "-min_improvement <float> :",  "specify the minimum improvement to stop RNNLM training (default: 1.003)");
    printf ("%35s\t%s\n", "-nglmstfile  <string>    :",     "specify the ngram lm stream file for interpolation");
    printf ("%35s\t%s\n", "-nsample     <int>       :",        "specify number of sample word from RNNLM (default: 1000)");
    printf ("%35s\t%s\n", "-unigramfile <string>    :",     "specify unigram lm file");
    printf ("%35s\t%s\n", "-sampletextfile <string> :",  "specify text file for sampling words from RNNLM");
    printf ("\nexample:\n");
    printf ("%s -train -trainfile data/train.dat -validfile data/dev.dat -device 1 -minibatch 64 -layers 31858:200i:400r:20002 -traincrit ce -lrtune newbob -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist  -debug 2 -randseed 1 -writemodel h200.mb64/rnnlm.txt -independent 1 -learnrate 1.0  -min_improvement 1.003\n", str);
    printf ("%s -ppl -readmodel h200.mb64/rnnlm.txt -testfile data/test.dat -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist -nglmstfile ng.st -lambda 0.5 -debug 2\n", str);
    printf ("%s -nbest -readmodel h200.mb64/rnnlm.txt.nbest -testfile data/test.dat  -inputwlist ./wlists/input.wlist -outputwlist ./wlists/output.wlist -nglmstfile ng.st -lambda 0.5 -debug 2\n", str);
}


bool isEmpty(string str)
{
    if (str == "EMPTY")     return true;
    else                    return false;
}

int string2int (string str)
{
    return atoi (str.c_str());
}

float string2float (string str)
{
    return atof (str.c_str());
}

void parseArray (string str, vector<string> &layersizes)
{
    int pos;
    int i, j;
    char numstr, typestr;
   layersizes.clear();
   while (str.size() > 0)
   {
       pos = str.find_first_of(':');
       if (pos ==  string::npos)        break;
       string substr = str.substr(0, pos);
       layersizes.push_back(substr);
       str = str.substr (pos+1);
   }
   layersizes.push_back(str);
}

float randomv(float min, float max)
{
#ifndef RAND48
    return rand()/(real)RAND_MAX*(max-min)+min;
#else
    return drand48()*(max-min)+min;
#endif
}

float gaussrandv(float mean, float var)
{
    float v1 = 0.0, v2 = 0.0, s = 0.0;
    int phase  = 0;
    double x;
    if (0 == phase)
    {
        do
        {
#ifndef RAND48
            float u1 = (float)rand()/RAND_MAX;
            float u2 = (float)rand()/RAND_MAX;
#else
            float u1 = drand48();
            float u2 = drand48();
#endif

            v1 = 2 * u1 - 1;
            v2 = 2 * u2 - 1;
            s = v1 * v1 + v2 * v2;
        } while ( 1 <= s || 0 == s);
        x = v1 * sqrt(-2 * log(s) / s);
    }
    else
    {
        x = v2 * sqrt(-2 * log(s) / s);
    }
    phase = 1 - phase;
    x = var*x+mean;
    return x;
}

int getline (char *line, int &max_words_line, FILE *&fptr)
{
    int i=0;
    char ch;
    while (!feof(fptr))
    {
        ch = fgetc(fptr);
        if (ch == ' ' && i==0)
        {
            continue;
        }
        line[i++] = ch;
        if (ch == '\n')
        {
            break;
        }
    }
    line[i] = 0;
    return i;
}

// log(exp(x) + exp(y))
float logadd (float x, float y)
{
    if (x > y)
    {
        return (x + log(1+exp(y-x)));
    }
    else
    {
        return (y + log(1+exp(x-y)));
    }
}




#endif
