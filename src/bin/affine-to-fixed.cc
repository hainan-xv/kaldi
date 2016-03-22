#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

using namespace std;

bool contains(string a, string b) {
  if (a.find(b) != string::npos) {
    return true;
  }
  return false;
}

int main() {
  string line;
  int in_final = 0;
  while (getline(cin, line)) {
    if (!contains(line, "NaturalGradientAffineComponent")) {
      cout << line << endl;
      continue;
    }
    // now we know it has it

    if (contains(line, "<NaturalGradientAffineComponent>")) {
      if (contains(line, "final")) {
        in_final = 1;
        cout << line << endl;
        continue;
      }
      stringstream ss(line);
      string s1, s2;
      ss >> s1 >> s2;
      cout << s1 << " " << s2 << " <FixedAffineComponent> <LinearParams>  [" << endl;
      continue;
    }
    assert(contains(line, "</NaturalGradientAffineComponent>"));
    if (in_final == 1) {
      in_final = 0;
      cout << line << endl;
    } else {
      cout << "</FixedAffineComponent>" << endl;
    }
  }
}

