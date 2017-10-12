#include "Mathops.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void Matrix::Init ()
{
    n_row = 0;
    n_col = 0;
    data = NULL;
    dataptr = NULL;
}

void Matrix::AllocMem ()
{
    if (n_row == 0 || n_col == 0)
    {
        cout << "ERROR: the number of row or column is 0 when trying to allocate memeory for matrx" << endl;
        exit (0);
    }
    if (data || dataptr)
    {
        cout << "Warning: the data and dataptr should be 0 before allocating memory" << endl;
    }
    ulint32 size_data = n_col * sizeof (int *);
    data = (int **) malloc (size_data);
    memset (data, 0, size_data);
    ulint32 size_dataptr = n_row * n_col * sizeof (int);
    dataptr = (int *) malloc (size_dataptr);
    memset (dataptr, 0, size_dataptr);
    for (ulint32 i = 0; i < n_col; i ++)
    {
        data[i] = dataptr + i * n_row;
    }
}

// initialize the Matrix
Matrix::Matrix ()
{
    Init ();
}
Matrix::Matrix (ulint32 nrow, ulint32 ncol)
{
    Init ();
    n_row = nrow;
    n_col = ncol;
    ulint32 size_data = n_col * sizeof (int *);
    data = (int **) malloc (size_data);
    memset (data,0, size_data);
    ulint32 size_dataptr = n_row * n_col * sizeof (int);
    dataptr = (int *)malloc (size_dataptr);
    memset (dataptr, 0, size_dataptr);
	for (ulint32 i = 0; i < n_col; i ++)
	{
		data[i] = dataptr + i * n_row;
	}
}
Matrix::Matrix (Matrix &mat, bool copy)
{
    Init ();
    dimension (mat.Getnrows() , mat.Getncols());
    AllocMem ();
    if (copy)
    {
        for (ulint32 i = 0; i < n_row; i ++)
        {
            for (ulint32 j = 0; j < n_col; j ++)
            {
                data[i][j] = mat[i][j];
            }
        }
    }
    else
    {
        data = mat.Getdatapoint ();
    }
}

Matrix::~Matrix ()
{
	if (data != NULL)
	{
        free (data);
        data = NULL;
	}
    if (dataptr != NULL)
    {
        free (dataptr);
        dataptr = NULL;
    }
}

void Matrix::freeMem ()   // free memory.
{
	if (data != NULL)
	{
        free (data);
        data = NULL;
	}
    if (dataptr != NULL)
    {
        free (dataptr);
        dataptr = NULL;
    }
}

void Matrix::dimension (ulint32 nrow, ulint32 ncol)
{
    n_row = nrow;
    n_col = ncol;
}

void Matrix::dump ()
{
    for (int i = 0; i < n_col; i ++)
    {
        printf ("col %d\t", i);
        for (int j = 0; j < n_row; j++ )
        {
            printf ("%d ", data[i][j]);
        }
        printf ("\n");
    }
}
