
#include <windows.h>

float func_4d(float x[4])
{    
    Sleep(500); //5 second pause
    return(x[0] - 3.5) * ((x[0] - 3.5)) + x[1] + x[2] + x[3];
}