#include "tree/event-map.h"
#include "tree/build-tree-utils.h"
#include "tree/build-tree-expand.h"

namespace kaldi {

vector<EventMap*> ExpandDecisionTree(const ContextDependency &ctx_dep,
                                     const BuildTreeStatsType &stats,
                                     const Questions &qo,
                                     int32 num_qst) {
  vector<BuildTreeStatsType> splits;
  SplitStatsByMap(stats, ctx_dep.ToPdfMap(), &splits);

  int num_leaves = ctx_dep.NumPdfs();
  KALDI_ASSERT(num_leaves == splits.size());

  std::vector<EventKeyType> all_keys;
  qo.GetKeysWithQuestions(&all_keys);

  int num_trees = num_qst;
  vector<EventMap*> ans(num_trees + 1, NULL); // 0 is for the original tree
  for (int i = 0; i < num_trees + 1; i++) {
    ans[i] = ctx_dep.ToPdfMap().Copy();
  }

  vector<vector<KeyYesset> > questions_for_trees(num_trees,
                                                 vector<KeyYesset>(num_leaves));
  // questions_for_trees[i][j] would be KeyYesset for i'th tree, j'th leaf
  // i'th tree means i'th among the ones that we are building

  if (all_keys.size() == 0) {
    KALDI_ERR << "ExpandDecisionTree(), no keys available to split "
     " on (maybe no key covered all of your events, or there was a problem"
     " with your questions configuration?)";
  }

  for (int l = 0; l < num_leaves; l++) {
    if (splits[l].size() == 0) {
      KALDI_ERR << "eh oh bad";
    }
    // process stats mapped to the l'th leaf, i.e. in splits[l]
    vector<KeyYesset> key_yesset_vec;

    for (size_t i = 0; i < all_keys.size(); i++) {
      if (qo.HasQuestionsForKey(all_keys[i])) {
        AppendNBestSplitsForKey(num_qst, splits[l], qo,
                                all_keys[i], &key_yesset_vec);
      }
    }

    sort(key_yesset_vec.begin(), key_yesset_vec.end());
    if (key_yesset_vec.size() > num_qst) {
      key_yesset_vec.resize(num_qst);
    }

    for (int j = 0; j < key_yesset_vec.size(); j++) {
      questions_for_trees[j][l] = key_yesset_vec[j];
    }
  }

  for (int j = 0; j < questions_for_trees.size(); j++) {
    for (int l = 0; l < num_leaves; l++) {
/*
      if (!(questions_for_trees[j][l].improvement > 0 ||
                   questions_for_trees[j][l].key == KeyYesset::NO_KEY)) {
        KALDI_LOG << questions_for_trees[j][l].improvement << " "
                  << questions_for_trees[j][l].key;
      }
//*/
      KALDI_ASSERT(questions_for_trees[j][l].improvement > 0 ||
                   questions_for_trees[j][l].key == KeyYesset::NO_KEY);
/*
     if (!(questions_for_trees[j][l].improvement > 0 ||
                   questions_for_trees[j][l].key == KeyYesset::NO_KEY)) {
        int kkk = 123;
      }
//      */
    }
  }

  for (int i = 0; i < num_trees; i++) {
    EventAnswerType next = num_leaves;
    KALDI_ASSERT(questions_for_trees[i].size() == num_leaves);
    ans[i + 1]->ExpandTree(questions_for_trees[i], &next);

    vector<BuildTreeStatsType> splits;
    SplitStatsByMap(stats, *ans[i + 1], &splits);

    KALDI_LOG << i + 1 << "'the tree, size is " << splits.size();
    for (int ii = 0; ii < splits.size(); ii++) {
      if (splits[ii].size() == 0) {
        KALDI_ERR << "This is bad!   "
                  << i + 1 << "'th tree, "
                  << ii << "'th leaf";

      }
    }
    KALDI_LOG << "Tree " << i + 1 << " is good";
  }
  return ans;
}

}  // namespace kaldi
