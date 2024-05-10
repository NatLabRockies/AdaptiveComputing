
#include <windows.h>
#include <cmath>

float func_4d(float x[4])
{    
    Sleep(500); //5 second pause
    //return sin ((x[0] - 3.5) * ((x[0] - 3.5)) + x[1] + x[2] + x[3]);
    return(x[0] - 3.5) * ((x[0] - 3.5)) + x[1] + x[2] + x[3];
}