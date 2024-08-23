#include <chrono>
#include <thread>

float* func_4d(float x[8])
{
    std::this_thread::sleep_for(std::chrono::milliseconds(5000)); // 5000 milliseconds = 5 seconds

    // Dynamically allocate memory for the result
    float* result = new float[6];
    
    result[0] = (x[0] - 3.5) * ((x[0] - 3.5)) + x[1] + x[2] + x[3];
    result[1] = (x[6] + x[4]) * (x[0] - x[7]);
    result[2] = (x[5] + x[4]) * (x[0] + x[2] - x[1]);
    result[3] = (x[6] - x[1] + x[4]) * (x[0] - x[2]);
    result[4] = (x[5] - x[4] * x[3] + x[2]) * (x[0] + x[3]);
    result[5] = (x[5] + x[6]) * (x[0] - x[4]);

    return result;
}

