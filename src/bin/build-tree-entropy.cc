// bin/build-tree-entropy.cc

// Copyright 2015 Hainan Xu

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep-multi.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train multiple decision trees\n"
        "Usage:  build-tree-entropy [options] <tree-stats-in> <roots-file> "
        " <questions-file> <topo-file> <tree-out-prefix> <virtual-tree>\n"
        "e.g.: \n"
        " build-tree num-trees=2 treeacc roots.txt 1.qst topo tree vtree\n";

    bool binary = true;
    int32 P = 1, N = 3;

    BaseFloat thresh = 300.0;

    // negative means use smallest split in splitting phase as thresh.
    BaseFloat cluster_thresh = -1.0;
    int32 max_leaves = 0;
    int32 num_trees = 1;
    double lambda = 0.0;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("context-width", &N, "Context window size [must match "
                "acc-tree-stats]");
    po.Register("central-position", &P, "Central position in context window "
                "[must match acc-tree-stats]");
    po.Register("max-leaves", &max_leaves, "Maximum number of leaves to be "
                "used in tree-buliding (if positive)");
    po.Register("thresh", &thresh, "Log-likelihood change threshold for "
                "tree-building");
    po.Register("cluster-thresh", &cluster_thresh, "Log-likelihood change "
                "threshold for clustering after tree-building.  0 means "
                "no clustering; -1 means use as a clustering threshold the "
                "likelihood change of the final split.");
    po.Register("lambda", &lambda, "weight for entropy term, default = 0");
    po.Register("num-trees", &num_trees,
                "number of trees to build, default = 1");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_filename = po.GetArg(1),
        roots_filename = po.GetArg(2),
        questions_filename = po.GetArg(3),
        topo_filename = po.GetArg(4),
        tree_out_filename = po.GetArg(5),
        virtual_tree_filename = po.GetArg(6),
        mapping_filename = po.GetArg(7);

    std::vector<std::vector<int32> > phone_sets;
    std::vector<bool> is_shared_root;
    std::vector<bool> is_split_root;
    {
      Input ki(roots_filename.c_str());
      ReadRootsFile(ki.Stream(), &phone_sets, &is_shared_root, &is_split_root);
    }

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }

    KALDI_LOG << "Number of separate statistics is " << stats.size();

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

    std::vector<int32> phone2num_pdf_classes;
    topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

    std::vector<const EventMap*> to_pdf_vec;

    //////// Build the tree. ////////////

    to_pdf_vec = BuildTreeMulti(qo,
                                phone_sets,
                                phone2num_pdf_classes,
                                is_shared_root,
                                is_split_root,
                                stats,
                                thresh,
                                max_leaves,
                                cluster_thresh,
                                P,
                                num_trees,
                                lambda);

    { // This block is to warn about low counts.
      for (size_t j = 0; j < num_trees; j++) {
        std::vector<BuildTreeStatsType> split_stats;
        SplitStatsByMap(stats, *to_pdf_vec[j],
                        &split_stats);
        for (size_t i = 0; i < split_stats.size(); i++)
          if (SumNormalizer(split_stats[i]) < 100.0)
            KALDI_VLOG(1) << "For tree-id " << j << " "
                          << "For pdf-id " << i << ", low count "
                          << SumNormalizer(split_stats[i]);
      }
    }

/*
    for (size_t j = 0; j < num_trees; j++) {
      ContextDependency ctx_dep(N, P, to_pdf_vec[j]);  // takes ownership
    // of pointer "to_pdf", so set it NULL.
      to_pdf_vec[j] = NULL;
      char temp[4];
      sprintf(temp, "-%d", (int)j);
      std::string tree_affix(temp);
      // tree files are like tree-2
      WriteKaldiObject(ctx_dep, tree_out_filename+tree_affix, binary);
    }
*/
    ContextDependencyMulti ctx_dep(N, P, to_pdf_vec, topo);

    for (size_t j = 0; j < num_trees; j++) {
    // of pointer "to_pdf", so set it NULL.
      to_pdf_vec[j] = NULL;
      char temp[4];
      sprintf(temp, "-%d", (int)j);
      std::string tree_affix(temp);
      // tree files are like tree-2
      ContextDependency c(N, P, ctx_dep.GetTree(j)); // take ownership
      WriteKaldiObject(c, tree_out_filename+tree_affix, binary);
    }

    EventMap* vtree;
    unordered_map<int32, vector<int32> > mapping;
    ctx_dep.GetVirtualTreeAndMapping(&vtree, &mapping);
    ContextDependency c(N, P, vtree);
    WriteKaldiObject(c, virtual_tree_filename, binary);


    {  // This block is just doing some checks.
      std::vector<int32> all_phones;
      for (size_t i = 0; i < phone_sets.size(); i++)
        all_phones.insert(all_phones.end(),
                          phone_sets[i].begin(), phone_sets[i].end());
      SortAndUniq(&all_phones);
      if (all_phones != topo.GetPhones()) {
        std::ostringstream ss;
        WriteIntegerVector(ss, false, all_phones);
        ss << " vs. ";
        WriteIntegerVector(ss, false, topo.GetPhones());
        KALDI_WARN << "Mismatch between phone sets provided in roots file,"
                   << " and those in topology: "
                   << ss.str();
      }
      std::vector<int32> phones_vec;  // phones we saw.
      PossibleValues(P, stats, &phones_vec);  // function in build-tree-utils.h

      std::vector<int32> unseen_phones;  // diagnostic.
      for (size_t i = 0; i < all_phones.size(); i++)
        if (!std::binary_search(phones_vec.begin(), phones_vec.end(),
                                all_phones[i]))
          unseen_phones.push_back(all_phones[i]);
      for (size_t i = 0; i < phones_vec.size(); i++)
        if (!std::binary_search(all_phones.begin(), all_phones.end(),
                                phones_vec[i]))
          KALDI_ERR << "Phone " << (phones_vec[i])
                    << " appears in stats but is not listed in roots file.";
      if (!unseen_phones.empty()) {
        std::ostringstream ss;
        for (size_t i = 0; i < unseen_phones.size(); i++)
          ss << unseen_phones[i] << ' ';
        // Note, unseen phones is just a warning as in certain kinds of
        // systems, this can be OK (e.g. where phone encodes position and
        // stress information).
        KALDI_WARN << "Saw no stats for following phones: " << ss.str();
      }
    }

    KALDI_LOG << "Wrote tree";

    DeleteBuildTreeStats(&stats);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
