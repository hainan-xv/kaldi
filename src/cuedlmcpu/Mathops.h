#ifndef _MATHOPS_H__
#define _MATHOPS_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "DataType.h"
using namespace std;

class Matrix
{
	// number of row and column
	ulint32 n_row, n_col;
	// data point
	int **data;
    int *dataptr;   // real date point
    void dimension (ulint32 nrow, ulint32 ncol);
    void Init ();
public:
	Matrix ();
	Matrix (ulint32 nrow, ulint32 ncol);
    Matrix (Matrix &mat, bool copy = false);
    void AllocMem ();
	~Matrix ();
    void freeMem ();
	inline ulint32 Getnrows () { return n_row; }
	inline ulint32 Getncols () { return n_col; }
	int *operator [] (ulint32 icol) { return data[icol]; }
	const int* operator [] (ulint32 icol) const { return data[icol]; }
    int ** Getdatapoint () { return data;}
    void dump ();    // dump the whole matrix
};
#endif
