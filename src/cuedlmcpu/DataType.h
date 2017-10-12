#include <time.h>
#include <iostream>
#ifndef _DATATYPE_DEF_H__
#define _DATATYPE_DEF_H__
typedef  short int          int16;
typedef unsigned short int  uint16;
// typedef long int            int32;
typedef unsigned long int   ulint32;

// dropout apply in the train time, but don't affect test time (better than standard)
// other, standard dropout
#define DROPOUT_V2

#define CUED_RESETVALUE 0.1

// variational dropout proposed by Y. Gal
// otherwise, used the dropout proposed by W. Zaremba
// #define VARIATIONALDROPOUT

// dropout the input word
// #define DROPOUTINWORD

// simple update input layer
#define SIMPLEUPDATEINPUTLYAER

#define INVALID_INT         1e8
#define MAX_STRING          100

// used for large train corpus, no load in one time
// #define CACHESIZE 1000

#define NUM_THREAD 1
// #define PRINTWENTROPY
// #define LOGLINEARINT

// initial value for recurrent state
#define RESETVALUE 0.1

typedef float real;

#define isdouble(real)  (sizeof(real) == sizeof(double))

// use momentum (not necessary from current experimental results)
// #define MOMENTUM

// relu ratio for positive input, found useful during training
#define RELURATIO 0.5

// Random initialization velue for weight
#define     MINRANDINITVALUE    -0.1
#define     MAXRANDINITVALUE    0.1


#define foreach_row(_i,_m)    for (size_t _i = 0; _i < (_m)->rows(); _i++)
#define foreach_column(_j,_m) for (size_t _j = 0; _j < (_m)->cols(); _j++)
#define foreach_coord(_i,_j,_m) for (size_t _j = 0; _j < (_m)->cols(); _j++) for (size_t _i = 0; _i < (_m)->rows(); _i++)

#ifdef _MSC_VER
#include <Windows.h>
/*
* from http://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows
*/



// timer class
struct timespec
{
    long tv_sec;
    long tv_nsec;
};    //header part

#define CLOCK_REALTIME 1

inline int clock_gettime(int, struct timespec *spec)      //C-file part
{  __int64 wintime; GetSystemTimeAsFileTime((FILETIME*)&wintime);
    wintime      -=116444736000000000i64;  //1jan1601 to 1jan1970
    spec->tv_sec  =wintime / 10000000i64;           //seconds
    spec->tv_nsec =wintime % 10000000i64 *100;      //nano-seconds
    return 0;
}
#endif

class auto_timer
{
    timespec time_start, time_end;
    real sec;
    real nsec;
    real acctime;
public:
    void start ()
    {
        clock_gettime (CLOCK_REALTIME, &time_start);
    }
    void end()
    {
         clock_gettime (CLOCK_REALTIME, &time_end);
    }
    void add()
    {
        end();
        if (time_end.tv_nsec - time_start.tv_nsec < 0)
        {
            nsec = 1000000000 + time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec - 1;
        }
        else
        {
            nsec = time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec;
        }
        acctime += sec + nsec * 1.0 / 1000000000;
    }
    real stop()
    {
        end();
        if (time_end.tv_nsec - time_start.tv_nsec < 0)
        {
            nsec = 1000000000 + time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec - 1;
        }
        else
        {
            nsec = time_end.tv_nsec - time_start.tv_nsec;
            sec = time_end.tv_sec - time_start.tv_sec;
        }
        return sec + nsec * 1.0/1000000000;
    }
    real getacctime ()
    {
        return acctime;
    }
    void clear()
    {
        sec = 0.0;
        nsec = 0.0;
        acctime = 0.0;
    }
    auto_timer ()
    {
        sec = 0.0;
        nsec = 0.0;
        acctime = 0.0;
        time_start.tv_sec = 0;
        time_end.tv_sec = 0;
        time_start.tv_nsec = 0;
        time_end.tv_nsec = 0;
    }
};
#endif
