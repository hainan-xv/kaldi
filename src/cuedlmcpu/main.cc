#include "helper.h"
#include "head.h"
#include "rnnlm.h"

using namespace std;

int main (int argc, char **argv)
{
    string str;
    string trainfile, validfile, testfile, feafile,
           traincrit, lrtune, inputwlist, outputwlist,
           nglmstfile, inmodelname, outmodelname, unigramfile,
           nceunigramfile, sampletextfile;
    int device, minibatch, chunksize, k, independent,
            cachesize, nsample, debug, rand_seed, fullvocsize,
            nthread, nclass, ptrnepoch;
    vector<string> layersizes;
    float lambda, learnrate, momentum, vrpenalty, min_improvement, lognormconst,
          gradient_cutoff, l2reg, diaginit, dropoutrate, lmscale,
          ip, ptrnlearnrate, nceunigramintpltwgt;
    bool binformat, flag_feature = false, flag_nceunigram = false;
	if (argc < 2)
	{
		printusage (argv[0]);
        return SUCCESS;
	}
	arguments arg (argc, argv);
	if (!arg.empty())
    {
        trainfile = arg.find("-trainfile");
        validfile = arg.find("-validfile");
        testfile = arg.find("-testfile");

        feafile = arg.find("-feafile");
        if (!isEmpty(feafile))                  flag_feature = true;
        else                                    flag_feature = false;

        str = arg.find("-device");
        if (!isEmpty(str))                      device = string2int(str);
        else                                    device = 0;             // default device value: 0

        str = arg.find("-minibatch");
        if (!isEmpty(str))                      minibatch = string2int(str);
         else                                   minibatch = 32;         // default minibatch value: 32
        assert (minibatch > 0);

        str = arg.find("-chunksize");
        if (!isEmpty(str))                      chunksize = string2int(str);
         else                                   chunksize = 1;         // default chunksize value: 1
        assert (chunksize > 0);

        str = arg.find("-layers");
        if (!isEmpty(str))                      parseArray(str, layersizes);
        // else                                    return ARGSPARSEERROR;

        str = arg.find("-dropout");
        if (!isEmpty(str))                      dropoutrate = string2float(str);
        else                                    dropoutrate = 0;

        str = arg.find("-diaginit");
        if (!isEmpty(str))                      diaginit = string2float(str);
        else                                    diaginit = -1;

        str = arg.find("-clipping");
        if (!isEmpty(str))                      gradient_cutoff = string2float(str);
        else                                    gradient_cutoff = 5;

        str = arg.find("-l2reg");
        if (!isEmpty(str))                      l2reg = string2float (str);
        else                                    l2reg = 0.0;

        str = arg.find("-traincrit");
        if (!isEmpty(str))                      traincrit = str;
        else                                    traincrit = "ce";

        str = arg.find("-lrtune");
        if (!isEmpty(str))                      lrtune = str;
        else                                    lrtune = "newbob";

        inputwlist = arg.find("-inputwlist");
        outputwlist = arg.find("-outputwlist");

        str = arg.find("-learnrate");
        if (!isEmpty(str))                      learnrate = string2float (str);
        else                                    learnrate = 0.8;    // default learnrate value: 1.0 (for mbsize = 32)

        str = arg.find ("-momentum");
        if (!isEmpty(str))                      momentum = string2float (str);
        else                                    momentum = 0.0;    // default momentum value: 0.0

        str = arg.find("-vrpenalty");
        if (!isEmpty(str))                      vrpenalty = string2float(str);
        else                                    vrpenalty = 0.0;    // default vrpenalty value: 0.0 (i.e. no variance regularization)

        str = arg.find("-ncesample");
        if (!isEmpty(str))                      k = string2int(str);
        else                                    k = 10;

        str = arg.find("-nclass");
        if (!isEmpty(str))                      nclass = string2int(str);
        else                                    nclass = 0;         // default nclass value: 0 (use full output layer)

        str = arg.find("-lognormconst");
        if (!isEmpty(str))                      lognormconst = string2float(str);
        else                                    lognormconst = -100.0;            // default lognormconst value: -100.0(a random value less than -10).

        str = arg.find("-lambda");
        if (!isEmpty(str))                      lambda = string2float(str);
        else                                    lambda = 0.5;

        str = arg.find("-cachesize");
        if (!isEmpty(str))                       cachesize = string2int(str);
        else                                     cachesize = 0;     // default cachesize value: 0 (don't use cache)

        str = arg.find("-debug");
        if (!isEmpty(str))                      debug = string2int(str);
        else                                    debug = 1;          // default debug value: 1

        str = arg.find("-nthread");
        if (!isEmpty(str))                      nthread = string2int(str);
        else                                    nthread = 1;          // default nthread value: 1

        str = arg.find("-randseed");
        if (!isEmpty(str))                      rand_seed = string2int(str);
        else                                    rand_seed = 1;      // default rand seed value: 1

        inmodelname  = arg.find("-readmodel");
        outmodelname = arg.find("-writemodel");

        str = arg.find("-fullvocsize");
        if (!isEmpty(str))                      fullvocsize = string2int(str);
        else                                    fullvocsize = 0;

        str = arg.find("-lmscale");
        if (!isEmpty(str))                      lmscale = string2float(str);
        else                                    lmscale = 12.0;

        str = arg.find("-ip");
        if (!isEmpty(str))                      ip = string2float(str);
        else                                    ip = 0.0;

        // ip = -10.0;

        str = arg.find("-independent");
        if (!isEmpty(str))                      independent = string2int(str);
        else                                    independent = 1;    // default: independent mode

        str = arg.find("-binformat");
        if (!isEmpty(str))                      binformat = true;
        else                                    binformat = false;  // default model format: TEXT

        str = arg.find("-min_improvement");
        if (!isEmpty(str))                      min_improvement = string2float(str);
        else                                    min_improvement = 1.003; //default min_improvement value: 1.001

        str = arg.find("-nglmstfile");
        if (!isEmpty(str))                      nglmstfile = str;
        else                                    lambda = 1.0;

        str = arg.find("-nsample");
        if (!isEmpty(str))                      nsample = string2int (str);
        else                                    nsample = 1000;         // default nsample value: 1000


        str = arg.find("-nceunigramfile");
        if (!isEmpty(str))
        {
            flag_nceunigram = true;
            nceunigramfile = str;
        }
        str = arg.find("-nceunigramintpltwgt");
        {
            if (!isEmpty(str))                  nceunigramintpltwgt = string2float(str);
            else                                nceunigramintpltwgt = 0.5;
        }

        str = arg.find("-unigramfile");
        if (!isEmpty(str))                      unigramfile = str;

        str = arg.find("-sampletextfile");
        if (!isEmpty(str))                      sampletextfile = str;

        if (debug > 1)
        {
            for (int i=0; i<argc; i++) printf ("%s ", argv[i]);
            printf ("\n");
        }

        if (!isEmpty(arg.find ("-ppl")))
        {
            RNNLM rnnlm (inmodelname, inputwlist, outputwlist, fullvocsize, binformat, debug);
            rnnlm.setNthread(nthread);
            rnnlm.calppl(testfile, lambda, nglmstfile);
        }
        else if (!isEmpty(arg.find ("-nbest")))
        {
            RNNLM rnnlm (inmodelname, inputwlist, outputwlist, fullvocsize, binformat, debug);
            rnnlm.setNthread (nthread);
            rnnlm.setLmscale (lmscale);
            rnnlm.setIp (ip);
            rnnlm.calnbest(testfile, lambda, nglmstfile);
        }
        else
        {
            printusage(argv[0]);
        }
    }
    else
    {
        printusage(argv[0]);
    }
}
