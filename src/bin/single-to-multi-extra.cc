#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <stdlib.h>
using namespace std;

vector<vector<double> > Get2dVectors(string filename, int n) {
  vector<vector<double> > ans;
  for (int i = 0; i < n; i++) {
    vector<double> v;
    ifstream ifile((filename + char('0' + i)).c_str());
    double p;
    int c = 0;
    while (ifile >> p) {
      v.push_back(p);
      c++;
    }
    cout << "# dimension is " << c << endl;
    ans.push_back(v);
  }
  return ans;
}
  

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
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == ',') s[i] = ' ';
  }
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
  if (argc != 6) {
    cout << "#" << argv[0] << " input_txt_file num_outputs leaves_string scale-file-prefix prior-file-prefix" << endl;
    cout << "you provided " << argc << "arguments:" << endl;
    for (int i = 1; i < argc; i++) {
      cout << i << ": " << argv[i] << endl;
    }
    exit(-1);
  }
  ifstream ifile(argv[1]);
  int num_outputs = 0;
  stringstream ss(argv[2]);
  ss >> num_outputs;
  string line;
  getline(ifile, line);
  assert(line == "<TransitionModel> ");

  vector<int> num_leaves = GetLeaves(argv[3]);
  vector<vector<double> > fixed_scale_vec = Get2dVectors(argv[4], num_outputs);
  vector<vector<double> > priors_vec = Get2dVectors(argv[5], num_outputs);

  do {
    getline(ifile, line);
  }
  while (line != "</TransitionModel> ");

  cout << "#finish reading transition model" << endl;
    
  vector<bool> add_extra(num_outputs, true);
  while (getline(ifile, line)) {
    if (line.find("output") != string::npos) {
      for (int i = 0; i < num_outputs; i++) {
//        string line_new = ReplaceString(line, "final", "final" + ('0' + i));
        string line_new = ReplaceString(line, "final", string("final") + char('0' + i));
        line_new = ReplaceString(line_new, "output ", string("output") + char('0' + i) + " ");
        cout << line_new << endl;
      }
      break;
    }
    // component-node name=final0-fixed-scale component=final0-fixed-scale input=final0-affine
    else if (line.find("name=final-fixed-scale") != string::npos) {
      for (int i = 0; i < num_outputs; i++) {
        string l = "component-node name=final-fixed-scale component=final-fixed-scale input=final-extra-affine";
        cout << ReplaceString(l, "final", string("final") + char('0' + i)) << endl;
      }
    }
    else if (line.find("final") != string::npos) {
      for (int i = 0; i < num_outputs; i++) {
//        string line_new = ReplaceString(line, "final", "final" + ('0' + i));
        string line_new = ReplaceString(line, "final", string("final") + char('0' + i));
        cout << line_new << endl;

        if (add_extra[i]) {
          // THE EXTRA IS HERE
          string final_nonlin = "component-node name=final-nonlin component=final-nonlin input=final-affine";
          cout << ReplaceString(final_nonlin, "final", string("final") + char('0' + i)) << endl;
          string final_renorm = "component-node name=final-renorm component=final-renorm input=final-nonlin";
          cout << ReplaceString(final_renorm, "final", string("final") + char('0' + i)) << endl;
          string final_extra_affine = "component-node name=final-extra-affine component=final-extra-affine input=final-renorm";
          cout << ReplaceString(final_extra_affine, "final", string("final") + char('0' + i)) << endl;
          add_extra[i] = false;
        }
      }
    }
    else {
      cout << line << endl;
    }
  }
  cout << "#finish reading topology file" << endl;

  while (getline(ifile, line)) {
    if (line.find("Prior") != string::npos) {
      break;
    }
    if (line.find("NumComponents") != string::npos) {
      stringstream ss(line);
      string s;
      int t;
      ss >> s >> t;
      cout << s << " " << t + num_outputs * 6 - 3 << endl;
    }
    else if (line.find("final") == string::npos) {
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
      getline(ifile, line);
      assert(IsPrefix("<Count> 0 </LogSoftmaxComponent>", line));

      for (int i = 0; i < num_outputs; i++) {
        // this line should read like "<ComponentName> final-affine <NaturalGradientAffineComponent> <LearningRate> 0.009239559 <LinearParams>  ["
        string line_new = ReplaceString(component_line, "final", string("final") + char('0' + i));
        cout << line_new << endl;
        for (int k = 0; k < 10 * input_dim; k++) {
          for (int j = 0; j < input_dim; j++) {
            cout << ((double(rand()) / RAND_MAX) - 0.5) * 0.0001 << " "; // might need to change this
          }
          if (k == 10 * input_dim - 1) {
            cout << " ] ";
          }
          cout << endl;
        }
        cout << "<BiasParams> [ ";
        for (int j = 0; j < 10 * input_dim; j++) {
          cout << ((double(rand()) / RAND_MAX) - 0.5) * 0.0001 << " "; // might need to change this
        }
        cout << " ] " << endl;
        cout << rand_line << endl; // this also ends the </NaturalGradientAffineComponent>
        // end of natural gradient component
        // EXTRA STUFF HERE
        cout << "<ComponentName> final" << i << "-nonlin <PnormComponent> <InputDim> "
             << input_dim * 10 << " <OutputDim> "
             << input_dim << " </PnormComponent>" << endl;

        cout << "<ComponentName> final" << i << "-renorm <NormalizeComponent> <InputDim> "
             << input_dim
             << " <TargetRms> 1 <AddLogStddev> F </NormalizeComponent>" << endl;

        line_new = ReplaceString(component_line, "final", string("final") + char('0' + i) + "-extra");
        cout << line_new << endl;
        for (int k = 0; k < num_leaves[i]; k++) {
          for (int j = 0; j < input_dim; j++) {
            cout << ((double(rand()) / RAND_MAX) - 0.5) * 0.0001 << " "; // might need to change this
          }
          if (k == num_leaves[i] - 1) {
            cout << " ] ";
          }
          cout << endl;
        }
        cout << "<BiasParams> [ ";
        for (int j = 0; j < num_leaves[i]; j++) {
          cout << ((double(rand()) / RAND_MAX) - 0.5) * 0.0001 << " "; // might need to change this
        }
        cout << " ] " << endl;
        cout << rand_line << endl; // this also ends the </NaturalGradientAffineComponent>




        cout << "<ComponentName> final" << i << "-fixed-scale <FixedScaleComponent> <Scales> [ "; // << endl;
//*
        assert(num_leaves[i] == fixed_scale_vec[i].size());
        for (int j = 0; j < fixed_scale_vec[i].size(); j++) {
          cout << fixed_scale_vec[i][j] << " ";
        }
// */
        cout << " ] " << endl;
        cout << "</FixedScaleComponent>" << endl;
        cout << "<ComponentName> final" << i << "-log-softmax <LogSoftmaxComponent> <Dim> " << num_leaves[i] << " <ValueAvg> [ ]" << endl;
        cout << "<DerivAvg>  [ ] " << endl;
        cout << "<Count> 0 </LogSoftmaxComponent>" << endl;

      }
    }
  }
  cout << "# now the priors_vec" << endl;
  assert(IsPrefix("</Nnet3> <LeftContext>", line));
  {
    stringstream ss(line);
    string s;
    for (int i = 0; i < 6; i++) {
      string tmp;
      ss >> tmp;
      s += tmp + " ";
    }
    cout << s << " ";
  }
  cout << num_outputs << " ";
  for (int i = 0; i < num_outputs; i++) {
    cout << " [ ";
    for (int j = 0; j < priors_vec[i].size(); j++) {
      cout << priors_vec[i][j] << " ";
    }
    cout << " ] " << endl;
  }
}
