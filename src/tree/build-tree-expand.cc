#include "tree/event-map.h"
#include "tree/build-tree-utils.h"
#include "tree/build-tree-expand.h"

namespace kaldi {

vector<EventMap*> ExpandDecisionTree(const ContextDependency &ctx_dep,
                                     const BuildTreeStatsType &stats,
                                     const Questions &qo,
                                     int32 num_qst) {

  // int32 N = ctx_dep.ContextWidth();

  vector<BuildTreeStatsType> splits;
  SplitStatsByMap(stats, ctx_dep.ToPdfMap(), &splits);

  int num_leaves = ctx_dep.NumPdfs();
  KALDI_ASSERT(num_leaves == splits.size());

  std::vector<EventKeyType> all_keys;
  qo.GetKeysWithQuestions(&all_keys);

  int num_trees = num_qst * all_keys.size();
  vector<EventMap*> ans(num_trees + 1, NULL);
  for (int i = 0; i < num_trees + 1; i++) {
    ans[i] = ctx_dep.ToPdfMap().Copy();
  }


  vector<vector<KeyYesset> > questions_for_trees(num_trees,
                                                 vector<KeyYesset>(num_leaves));
  // questions_for_trees[i][j] would be KeyYesset for i'th tree, j'th leaf
  /*
  for (int i = 0; i < num_qst * all_keys.size(); i++) {
    vector<KeyYesset> v(num_leaves, KeyYesset());
    questions_for_trees.push_back(v);
  }
// */


  if (all_keys.size() == 0) {
    KALDI_ERR << "ExpandDecisionTree(), no keys available to split "
     " on (maybe no key covered all of your events, or there was a problem"
     " with your questions configuration?)";
  }

  for (int l = 0; l < num_leaves; l++) {
    // process stats mapped to the l'th leaf, i.e. in splits[l]
    vector<vector<EventValueType> > yes_set_vec;
    vector<BaseFloat> improvement_vec;
    vector<EventKeyType> keys;

    for (size_t i = 0; i < all_keys.size(); i++) {
      if (qo.HasQuestionsForKey(all_keys[i])) {
        FindNBestSplitsForKey(num_qst, splits[l], qo,
                              all_keys[i], &yes_set_vec, &improvement_vec);
        keys.resize(yes_set_vec.size(), all_keys[i]);
      }
    }
    for (int j = 0; j < yes_set_vec.size(); j++) {
      questions_for_trees[j][l] = KeyYesset(keys[j], yes_set_vec[j]);
    }
  }

  for (int i = 0; i < num_trees; i++) {
    EventAnswerType next = num_leaves;
    KALDI_ASSERT(questions_for_trees[i].size() == num_leaves);
    ans[i + 1]->ExpandTree(questions_for_trees[i], &next);
  }
  return ans;
}

}  // namespace kaldi
