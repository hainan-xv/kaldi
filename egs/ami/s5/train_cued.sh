

cued-rnnlm.v1.0/rnnlm.cued.v1.0 -train -trainfile data/rnnlm/text_nosp/ami.txt \
    -device `free-gpu` -inputwlist input.txt -outputwlist output.txt \
    -bptt 5 -traincrit ce -independent 1 -learnrate 1.0 \
    -validfile data/rnnlm/text_nosp/dev.txt -layers 10002:200:10002 -minibatch 64 -writemodel modrl/rnnlm.txt
