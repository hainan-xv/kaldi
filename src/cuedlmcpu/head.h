#ifndef __HEAD_H__
#define __HEAD_H__

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "math.h"

using namespace std;
#ifdef _MSC_VER
extern "C" {
    extern double drand48(void);
    extern long lrand48(void);
    extern void srand48(long);
}
#endif


#define     RAND48              // this is to initialize the weight using 48-bit random number generator (from Andreas's comment).
#define     SUCCESS             0
#define     FILEREADERROR       1
#define     ARGSPARSEERROR      2
#define     EOL                 -2
#define     CHECKNUM            9999999
#define     MAX_WORD_LINE       4008

typedef     map<string, int>    WORDMAP;
typedef     float               real;

// #define     DEBUG
#endif
