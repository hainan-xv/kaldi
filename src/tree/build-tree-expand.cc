#include "tree/event-map.h"
#include "tree/build-tree-utils.h"
#include "tree/build-tree-expand.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

vector<EventMap*> ExpandDecisionTree(const ContextDependency &ctx_dep,
                                     const BuildTreeStatsType &stats,
                                     const Questions &qo,
                                     int32 num_qst) {
  vector<BuildTreeStatsType> splits;
//  KALDI_LOG << "stats size is " << splits.size();

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
    KALDI_LOG << "For leaf " << l << " stats size is " << splits[l].size();
    // process stats mapped to the l'th leaf, i.e. in splits[l]
    vector<KeyYesset> key_yesset_vec;

    for (size_t i = 0; i < all_keys.size(); i++) {
      if (qo.HasQuestionsForKey(all_keys[i])) {
        int old_size = key_yesset_vec.size();
        AppendNBestSplitsForKey(num_qst, splits[l], qo,
                                all_keys[i], &key_yesset_vec);
        if (key_yesset_vec.size() - old_size == 1) {
          vector<EventValueType> yesset;
          BaseFloat im = FindBestSplitForKey(splits[l], qo, all_keys[i], &yesset);
//          KALDI_LOG << "__difference is " << im - key_yesset_vec[old_size].improvement
//            << im << "   " << key_yesset_vec[old_size].improvement;
//          KALDI_LOG << "__key is " << all_keys[i] << "   " << key_yesset_vec[old_size].key;
          KALDI_ASSERT(im == key_yesset_vec[old_size].improvement);
          KALDI_ASSERT(all_keys[i] == key_yesset_vec[old_size].key);

          // delete this part TODO(hxu)
          if (yesset != key_yesset_vec[old_size].yes_set) {
//            KALDI_LOG << "__yes set different";
            using std::cout;
            using std::endl;
/*            for (int jj = 0; jj < yesset.size(); jj++) {
              cout << yesset[jj] << " ";
            } cout << endl;
            for (int jj = 0; jj < key_yesset_vec[old_size].yes_set.size(); jj++) {
              cout << key_yesset_vec[old_size].yes_set[jj] << " ";
            } cout << endl; */
            key_yesset_vec[old_size].yes_set = yesset;
          }
        }
      }
    }

    sort(key_yesset_vec.begin(), key_yesset_vec.end());
    if (key_yesset_vec.size() > num_qst) {
      KALDI_LOG << l << " trim " << key_yesset_vec.size() << " to " << num_qst;
      key_yesset_vec.resize(num_qst);
    }

    for (int i = 0; i < key_yesset_vec.size(); i++) {
      KALDI_LOG << "improvement for " << i << " is "
                << key_yesset_vec[i].improvement
                << " key is " << key_yesset_vec[i].key;
    }

    for (int j = 0; j < key_yesset_vec.size(); j++) {
      questions_for_trees[j][l] = key_yesset_vec[j];
    }
    
    // TODO(hxu) not neccesary, for debug only...
    for (int j = key_yesset_vec.size(); j < num_qst; j++) {
      KALDI_LOG << "for tree " << j << " no question for leaf " << l;
      KALDI_ASSERT(questions_for_trees[j][l].key == KeyYesset::NO_KEY);
    }
    KALDI_LOG << "";
  }

  for (int j = 0; j < questions_for_trees.size(); j++) {
    for (int l = 0; l < num_leaves; l++) {
//*
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

  {
    vector<BuildTreeStatsType> local_splits;
    SplitStatsByMap(stats, *ans[0], &local_splits);

    KALDI_LOG << 0 << "'th tree, size is " << local_splits.size();
    for (int ii = 0; ii < local_splits.size(); ii++) {
//      KALDI_LOG << "expanded " << ii << " has " << local_splits[ii].size()
//                << " stats";
      if (local_splits[ii].size() == 0) {
        KALDI_ERR << "This is bad!   "
                  << 0 << "'th tree, "
                  << ii << "'th leaf";

      }
    }
    KALDI_LOG << "Tree " << 0 << " is good";
  }

  for (int i = 0; i < num_trees; i++) {
    EventAnswerType next = num_leaves;
    KALDI_ASSERT(questions_for_trees[i].size() == num_leaves);
    KALDI_LOG << i + 1 <<"'th tree, length of questions vec is " << questions_for_trees[i].size();

    for (int j = 0; j < questions_for_trees[i].size(); j++) {
      if (questions_for_trees[i][j].key != KeyYesset::NO_KEY) {
        ;
//        KALDI_LOG << "__good key: " << j;
      }
    }

    ans[i + 1]->ExpandTree(questions_for_trees[i], &next);

    vector<BuildTreeStatsType> local_splits;
    SplitStatsByMap(stats, *ans[i + 1], &local_splits);

    KALDI_LOG << i + 1 << "'the tree, size is " << local_splits.size();
    for (int ii = 0; ii < local_splits.size(); ii++) {
      BaseFloat count = 0.0;
      for (int jj = 0; jj < local_splits[ii].size(); jj++) {
        count += ((GaussClusterable*)(local_splits[ii][jj]).second)->count();
      }
      KALDI_LOG << "expanded leaf " << ii << " has " << local_splits[ii].size()
                << " stats; count is " << count;
      if (local_splits[ii].size() == 0) {
//        /*
        KALDI_ERR << "This is bad!   "
                  << i + 1 << "'th tree, "
                  << ii << "'th leaf";
//                  */


      }
    }
    KALDI_LOG << "Tree " << i + 1 << " is good";
  }
  return ans;
}

}  // namespace kaldi
