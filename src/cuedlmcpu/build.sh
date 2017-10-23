# make clean
rm -f *.o rnnlm.cued.v1.0.eval
g++ -g -std=c++0x main.cc rnnlm.cc  Mathops.cc fileops.cc helper.cc layer.cc  -o rnnlm.cued.v1.0.eval -lrt -fopenmp
