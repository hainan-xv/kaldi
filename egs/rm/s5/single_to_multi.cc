#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <stdlib.h>
using namespace std;

int NumFields(string s) {
  stringstream ss(s);
  int ans = 0;
  string str;
  while (ss >> str) {
    ans++;
  }
  return ans;
}

vector<int> GetLeaves(string s) {
  vector<int> ans;
  stringstream ss(s);
  int str;
  while (ss >> str) {
    ans.push_back(str);
  }
  return ans;
}

bool IsPrefix(string s1, string s2) {
  for (int i = 0; i < s1.size(); i++) {
    if (s1[i] != s2[i]) return false;
  }
  return true;
}

string ReplaceString(string subject, const string& search,
                          const string& replace) {
  size_t pos = 0;
  while((pos = subject.find(search, pos)) != string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
}

int main(int argc, char** argv) {
  cout << "#" << argv[0] << " input_txt_file num_outputs leaves_string";
  ifstream ifile(argv[1]);
  int num_outputs = 0;
  stringstream ss(argv[2]);
  ss >> num_outputs;
  string line;
  getline(ifile, line);
  assert(line == "<TransitionModel> ");

  vector<vector<double> > fixed_scale_vec;
  vector<int> num_leaves = GetLeaves(argv[3]);

  do {
    getline(ifile, line);
  }
  while (line != "</TransitionModel> ");

  cout << "#finish reading transition model" << endl;
    
  while (getline(ifile, line)) {
    if (line.find("output") != string::npos) {
      for (int i = 0; i < num_outputs; i++) {
//        string line_new = ReplaceString(line, "final", "final" + ('0' + i));
        string line_new = ReplaceString(line, "final", string("final") + char('0' + i));
        line_new = ReplaceString(line, "output ", string("output") + char('0' + i) + " ");
        cout << line_new << endl;
      }
      break;
    }
    else if (line.find("final") != string::npos) {
      for (int i = 0; i < num_outputs; i++) {
//        string line_new = ReplaceString(line, "final", "final" + ('0' + i));
        string line_new = ReplaceString(line, "final", string("final") + char('0' + i));
        cout << line_new << endl;
      }
    }
    else {
      cout << line << endl;
    }
  }
  cout << "#finish reading topology file" << endl;

  while (getline(ifile, line)) {
    if (line.find("final") == string::npos) {
      cout << line << endl;
    }
    else {
      // here we have the line as <ComponentName> final-affine <NaturalGradientAffineComponent> ...
      assert(IsPrefix("<ComponentName>", line));
      string component_line = line;
//      int num_rows = 0;
      getline(ifile, line);
      int input_dim = NumFields(line);
      do {
        getline(ifile, line);
//        num_rows++;
      } while (line[0] != '<');
//      num_rows--;
      // now the line is the <BiasParams> ..., one line
      cout << "# input-dim is " << input_dim << endl;
      assert(IsPrefix("<BiasParams>", line));
      getline(ifile, line); // this line is  <RankIn> 20 <RankOut> 80 <UpdatePeriod> ..., simply copy it over
      assert(IsPrefix("<RankIn>", line));
      string rand_line = line;

      getline(ifile, line); // this line is <ComponentName> final-fixed-scale <FixedScaleComponent> <Scales>
      assert(IsPrefix("<ComponentName> final-fixed-scale <FixedScaleComponent> <Scales>", line));
      getline(ifile, line);
      assert(IsPrefix("</FixedScaleComponent>", line));
      getline(ifile, line);
      assert(IsPrefix("<ComponentName> final-log-softmax <LogSoftmaxComponent>", line));
      getline(ifile, line);
      assert(IsPrefix("<DerivAvg>", line));

      for (int i = 0; i < num_outputs; i++) {
        // this line should read like "<ComponentName> final-affine <NaturalGradientAffineComponent> <LearningRate> 0.009239559 <LinearParams>  ["
        string line_new = ReplaceString(component_line, "final", string("final") + char('0' + i));
        for (int k = 0; k < num_leaves[i]; k++) {
          for (int j = 0; j < input_dim; j++) {
            cout << ((double(rand()) / RAND_MAX) - 0.5) * 0.0001 << " "; // might need to change this
          }
          if (k == - num_leaves[i] - 1) {
            cout << " ] ";
          }
          cout << endl;
          cout << "<BiasParams> [ ";
          for (int j = 0; j < input_dim; j++) {
            cout << ((double(rand()) / RAND_MAX) - 0.5) * 0.0001 << " "; // might need to change this
          }
          cout << " ] " << endl;
          cout << rand_line << endl; // this also ends the </NaturalGradientAffineComponent>
          // end of natural gradient component
          cout << "<ComponentName> final" << i << "-fixed-scale <FixedScaleComponent> <Scales> [ " << endl;
/*
          for (int j = 0; j < fixed_scale_vec[i].size(); j++) {
            cout << fixed_scale_vec[i][j] << " ";
          }
// */
          cout << " ] " << endl;
          cout << "</FixedScaleComponent>" << endl;
          cout << "<ComponentName> final" << i << "-log-softmax <LogSoftmaxComponent> <Dim> " << num_leaves[i] << " ValueAvg> [ ]" << endl;
          cout << "<DerivAvg>  [ ] ";

        }
      }


    }

  }
}
