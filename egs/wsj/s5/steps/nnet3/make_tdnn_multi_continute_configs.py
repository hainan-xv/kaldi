#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings



parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 0 -2,2 0 -4,4 0 -8,8'")
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--pnorm-input-dim", type=int,
                    help="input dimension to p-norm nonlinearities")
parser.add_argument("--pnorm-output-dim", type=int,
                    help="output dimension of p-norm nonlinearities")
parser.add_argument("--relu-dim", type=int,
                    help="dimension of ReLU nonlinearities")
parser.add_argument("--use-presoftmax-prior-scale", type=str,
                    help="if true, a presoftmax-prior-scale is added",
                    choices=['true', 'false'], default = "true")
parser.add_argument("--num-targets", type=int,
                    help="number of network targets (e.g. num-pdf-ids/num-leaves)")
parser.add_argument("config_dir",
                    help="Directory to write config files and variables");
parser.add_argument("--num-targets-multi", type=str,
                    help="number of network targets (e.g. num-pdf-ids/num-leaves)")
parser.add_argument("--objective", type=str,
                    help="objective type",
                    choices=['linear', 'quadratic'], default = "linear")

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.config_dir):
    os.makedirs(args.config_dir)

## Check arguments.
if args.splice_indexes is None:
    sys.exit("--splice-indexes argument is required");
if args.feat_dim is None or not (args.feat_dim > 0):
    sys.exit("--feat-dim argument is required");
#if args.num_targets is None or not (args.feat_dim > 0):
#    sys.exit("--feat-dim argument is required");  #TODO(hxu)

targets = args.num_targets_multi.split(",");
num_targets_list = [];
for i in targets:
  num_targets_list.append(int(i))

if not args.relu_dim is None:
    if not args.pnorm_input_dim is None or not args.pnorm_output_dim is None:
        sys.exit("--relu-dim argument not compatible with "
                 "--pnorm-input-dim or --pnorm-output-dim options");
    nonlin_input_dim = args.relu_dim
    nonlin_output_dim = args.relu_dim
else:
    if not args.pnorm_input_dim > 0 or not args.pnorm_output_dim > 0:
        sys.exit("--relu-dim not set, so expected --pnorm-input-dim and "
                 "--pnorm-output-dim to be provided.");
    nonlin_input_dim = args.pnorm_input_dim
    nonlin_output_dim = args.pnorm_output_dim

if args.use_presoftmax_prior_scale == "true":
    use_presoftmax_prior_scale = True
else:
    use_presoftmax_prior_scale = False

## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
splice_array = []
left_context = 0
right_context = 0
split1 = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split1) < 1:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split1:
        split2 = string.split(",")
        if len(split2) < 1:
            sys.exit("invalid --splice-indexes argument, too-short element: "
                     + args.splice_indexes)
        int_list = []
        for int_str in split2:
            int_list.append(int(int_str))
        if not int_list == sorted(int_list):
            sys.exit("elements of --splice-indexes must be sorted: "
                     + args.splice_indexes)
        left_context += -int_list[0]
        right_context += int_list[-1]
        splice_array.append(int_list)
except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + e)
left_context = max(0, left_context)
right_context = max(0, right_context)
num_hidden_layers = len(splice_array)
input_dim = len(splice_array[0]) * args.feat_dim  +  args.ivector_dim

f = open(args.config_dir + "/vars", "w")
print('left_context=' + str(left_context), file=f)
print('right_context=' + str(right_context), file=f)
# the initial l/r contexts are actually not needed.
# print('initial_left_context=' + str(splice_array[0][0]), file=f)
# print('initial_right_context=' + str(splice_array[0][-1]), file=f)
print('num_hidden_layers=' + str(num_hidden_layers), file=f)
f.close()

f = open(args.config_dir + "/init.config", "w")
print('# Config file for initializing neural network prior to', file=f)
print('# preconditioning matrix computation', file=f)
print('input-node name=input dim=' + str(args.feat_dim), file=f)
list=[ ('Offset(input, {0})'.format(n) if n != 0 else 'input' ) for n in splice_array[0] ]
if args.ivector_dim > 0:
    print('input-node name=ivector dim=' + str(args.ivector_dim), file=f)
    list.append('ReplaceIndex(ivector, t, 0)')
# example of next line:
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"
if args.objective == "linear":
    for i in range(0, len(targets)):
        print('output-node name=output' + str(i) + ' input=Append({0})'.format(", ".join(list)), file=f)
else:
    for i in range(0, len(targets)):
        print('output-node name=output' + str(i) + ' objective=quadratic input=Append({0})'.format(", ".join(list)), file=f)
    
f.close()

total_num_params = 0
for l in range(1, 2):
    f = open(args.config_dir + "/layer{0}.config".format(l), "w")
    print('# Config file for layer {0} of the network'.format(l), file=f)

    cur_dim = (nonlin_output_dim * len(splice_array[l-1]) if l > 1 else input_dim)

    nnn=0
    for i in num_targets_list:
        print('# Note: param-stddev in next component defaults to 1/sqrt(input-dim).', file=f)
        print('component name=affine{0}-{3} type=NaturalGradientAffineComponent '
              'input-dim={1} output-dim={2} bias-stddev=0'.
            format(l, cur_dim, nonlin_input_dim, nnn), file=f)
        total_num_params = total_num_params + cur_dim + nonlin_input_dim

        if args.relu_dim is not None: # TODO(hxu) not finished for here yet
            print('component name=nonlin{0}-{2} type=RectifiedLinearComponent dim={1}'.
                  format(l, args.relu_dim, nnn), file=f)
        else:
            print('# In nnet3 framework, p in P-norm is always 2.', file=f)
            print('component name=nonlin{0}-{3} type=PnormComponent input-dim={1} output-dim={2}'.
                  format(l, args.pnorm_input_dim, args.pnorm_output_dim, nnn), file=f)
        print('component name=renorm{0}-{2} type=NormalizeComponent dim={1}'.format(
             l, nonlin_output_dim, nnn), file=f)

        print('component name=final-affine' + str(nnn) + ' type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} param-stddev=0 bias-stddev=0'.format(
          nonlin_output_dim, i), file=f)
        total_num_params = total_num_params + nonlin_output_dim * i
        nnn = nnn + 1
    # printing out the next two, and their component-nodes, for l > 1 is not
    # really necessary as they will already exist, but it doesn't hurt and makes
    # the structure clearer.
    if use_presoftmax_prior_scale :
        for i in range(0, len(targets)):
            print('component name=final-fixed-scale' + str(i) + ' type=FixedScaleComponent '
              'scales={0}/presoftmax_prior_scale.vec'.format(
              args.config_dir) + str(i), file=f)

    nnn = 0
    for i in num_targets_list:
        print('component name=final-log-softmax' + str(nnn) + ' type=LogSoftmaxComponent dim={0}'.format(
          i), file=f)
        nnn = nnn + 1

    print('# Now for the network structure', file=f)
    nnn = 0
    for i in num_targets_list:
        # e.g. cur_input = 'Append(Offset(renorm1, -2), renorm1, Offset(renorm1, 2))'
        splices = [ ('Offset(renorm{0}, {1})'.format(l-1, n) if n !=0 else 'renorm{0}'.format(l-1))
                    for n in splice_array[l-1] ]
        cur_input='Append({0})'.format(', '.join(splices))

        print('component-node name=affine{0}-{2} component=affine{0}-{2} input={1} '.
              format(l, cur_input, nnn), file=f)
        print('component-node name=nonlin{0}-{1} component=nonlin{0}-{1} input=affine{0}-{1}'.
              format(l, nnn), file=f)
        print('component-node name=renorm{0}-{1} component=renorm{0}-{1} input=nonlin{0}-{1}'.
              format(l, nnn), file=f)
        nnn = nnn + 1

    for i in range(0, len(targets)):
        print('component-node name=final-affine' + str(i) + ' component=final-affine' + str(i) + ' input=renorm{0}-{1}'.
        format(l, i), file=f)
        if use_presoftmax_prior_scale:
            print('component-node name=final-fixed-scale' + str(i) + ' component=final-fixed-scale' + str(i) + ' input=final-affine' + str(i) + '',
              file=f)
            print('component-node name=final-log-softmax' + str(i) + ' component=final-log-softmax' + str(i) + ' '
              'input=final-fixed-scale' + str(i), file=f)
        else:
            print('component-node name=final-log-softmax' + str(i) + ' component=final-log-softmax' + str(i) + ' '
              'input=final-affine', file=f)

    for i in range(0, len(targets)):
        print('output-node name=output' + str(i) + ' input=final-log-softmax' + str(i), file=f)

    f.close()
print("total number of parameters is " + str(total_num_params))
# component name=nonlin1 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm1 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component name=final-log-softmax type=LogSoftmaxComponent dim=$num_leaves


# ## Write file $config_dir/init.config to initialize the network, prior to computing the LDA matrix.
# ##will look like this, if we have iVectors:
# input-node name=input dim=13
# input-node name=ivector dim=100
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"

# ## Write file $config_dir/layer1.config that adds the LDA matrix, assumed to be in the config directory as
# ## lda.mat, the first hidden layer, and the output layer.
# component name=lda type=FixedAffineComponent matrix=$config_dir/lda.mat
# component name=affine1 type=NaturalGradientAffineComponent input-dim=$lda_input_dim output-dim=$pnorm_input_dim bias-stddev=0
# component name=nonlin1 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm1 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component name=final-log-softmax type=LogSoftmax dim=$num_leaves
# # InputOf(output) says use the same Descriptor of the current "output" node.
# component-node name=lda component=lda input=InputOf(output)
# component-node name=affine1 component=affine1 input=lda
# component-node name=nonlin1 component=nonlin1 input=affine1
# component-node name=renorm1 component=renorm1 input=nonlin1
# component-node name=final-affine component=final-affine input=renorm1
# component-node name=final-log-softmax component=final-log-softmax input=final-affine
# output-node name=output input=final-log-softmax


# ## Write file $config_dir/layer2.config that adds the second hidden layer.
# component name=affine2 type=NaturalGradientAffineComponent input-dim=$lda_input_dim output-dim=$pnorm_input_dim bias-stddev=0
# component name=nonlin2 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm2 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component-node name=affine2 component=affine2 input=Append(Offset(renorm1, -2), Offset(renorm1, 2))
# component-node name=nonlin2 component=nonlin2 input=affine2
# component-node name=renorm2 component=renorm2 input=nonlin2
# component-node name=final-affine component=final-affine input=renorm2
# component-node name=final-log-softmax component=final-log-softmax input=final-affine
# output-node name=output input=final-log-softmax


# ## ... etc.  In this example it would go up to $config_dir/layer5.config.

