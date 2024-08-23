#include <thread>

float func_4d(float x[4])
{    
    std::this_thread::sleep_for(std::chrono::milliseconds(5000)); // 5000 milliseconds = 5 seconds
    return(x[0] - 3.5) * ((x[0] - 3.5)) + x[1] + x[2] + x[3];
}
