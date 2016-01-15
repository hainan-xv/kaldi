// bin/build-tree-virtual.cc

// Copyright 2014 Hainan XU

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/context-dep-multi.h"
#include "tree/build-tree.h"
#include "tree/build-tree-virtual.h"
#include "tree/build-tree-expand.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

using std::string;
using std::vector;
using std::pair;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Expand the leaves of the input decision tree, add N questions\n"
        "Usage:  build-tree-expand [options]"
        " <tree-in> <topo-file> <question-file> <stats> <matrix-out> <tree-out>\n";

    bool binary = true;
    int32 num_qst = 1;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-questions", &num_qst,
        "number of questions to expand the tree");

    po.Read(argc, argv);

    if (po.NumArgs() != 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_in = po.GetArg(1),
         topo_filename = po.GetArg(2),
         questions_filename = po.GetArg(3),
         stats_filename = po.GetArg(4),
         matrix_filename = po.GetArg(5),
         trees_out = po.GetArg(6),
         tree_out = po.GetArg(7),
         map_out = po.GetArg(8);

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    Questions qo;
    {
      bool binary_in;
      try {
        Input ki(questions_filename, &binary_in);
        qo.Read(ki.Stream(), binary_in);
      } catch (const std::exception &e) {
        KALDI_ERR << "Error reading questions file " << questions_filename
                  << ", error is: " << e.what();
      }
    }

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in, &ctx_dep);

    KALDI_LOG << "Building expanding trees... ";

    vector<EventMap*> out = 
         ExpandDecisionTree(ctx_dep, stats, qo, num_qst);

    int32 N = ctx_dep.ContextWidth(), P = ctx_dep.CentralPosition();
    vector<pair<int32, int32> > NPs(out.size(), std::make_pair(N, P));

    // pointer owned here
    ContextDependencyMulti ctx_dep_multi(NPs, out, topo);
    
    unordered_map<int32, vector<int32> > mappings;
    EventMap* merged_tree;
    ctx_dep_multi.GetVirtualTreeAndMapping(&merged_tree, &mappings);

    KALDI_LOG << "Number of virtual leaves: " << mappings.size();
    KALDI_LOG << "Number of leaves per tree before expanding: " << ctx_dep.NumPdfs();

    SparseMatrix<BaseFloat> matrix;
    ExpandedMappingToSparseMatrix(mappings, ctx_dep.NumPdfs(), &matrix);

    WriteKaldiObject(matrix, matrix_filename, binary);
//    WriteKaldiObject(ctx_dep_multi, tree_out, binary);
    {
      Output o(tree_out, binary);
      ctx_dep_multi.WriteVirtualTree(o.Stream(), binary);
    }

    {
      Output mapfile_output(map_out, binary);
      WriteMultiTreeMapping(mappings, mapfile_output.Stream(), binary, out.size());
    }

    for (size_t j = 0; j < out.size(); j++) {
      ContextDependency ctx_dep(N, P, out[j]->Copy());  // takes ownership
    // of pointer "to_pdf", so set it NULL.
      out[j] = NULL;
      char temp[4];
      sprintf(temp, "-%d", (int)(j));
      std::string tree_affix(temp);
      // tree files are like tree-2
      WriteKaldiObject(ctx_dep, trees_out+tree_affix, binary);
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
