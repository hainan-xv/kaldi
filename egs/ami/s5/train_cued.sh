
. path.sh
. cmd.sh



$cuda_cmd -l hostname=b01*  speed_test_nce/train_cued.log cued-rnnlm.v1.0/rnnlm.cued.v1.0 -train -trainfile data/rnnlm/text_nosp/ami.txt \
    -device `free-gpu` -inputwlist input.txt -outputwlist output.txt \
    -minibatch 64 -chunksize 6 \
    -traincrit nce \
    -randseed 1 \
    -bptt 5 -traincrit ce -independent 1 -learnrate 2.0 \
    -lrtune newbob \
    -independent 1 -learnrate 1.0 \
    -validfile data/rnnlm/text_nosp/dev.txt -layers 10002:200i:200m:10002 -minibatch 64 -writemodel speed_test_nce/rnnlm.txt

#cued-rnnlm.v1.0/rnnlm.cued.v1.0 -train -trainfile data/rnnlm/text_nosp/ami.txt \
#    -device `free-gpu` -inputwlist input.txt -outputwlist output.txt \
#    -minibatch 64 -chunksize 6 \
#    -traincrit ce \
#    -randseed 1 \
#    -bptt 5 -traincrit ce -independent 1 -learnrate 2.0 \
#    -lrtune newbob \
#    -independent 1 -learnrate 1.0 \
#    -validfile data/rnnlm/text_nosp/dev.txt -layers 10002:200i:200g:10002 -minibatch 64 -writemodel my_cued_model_200/rnnlm.txt

#./rnnlm.cued.v1.0 -train -trainfile data/train.dat -validfile data/dev.dat -device 1 -minibatch 64 -chunksize 6 -layers 9120:512i:512g:9120 -traincrit ce -lrtune newbob -inputwlist wlists/input.wlist.index -outputwlist wlists/input.wlist.index -debug 2 -randseed 1 -writemodel h512.mb64.chunk6.ce/rnnlm.txt -independent 1 -learnrate 1.0 
