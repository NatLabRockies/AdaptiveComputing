//"call_python.cpp -o output -L C:\Python39\libs -lpython39 -I C:\Python39\include

#include <python.h>
#include <iostream>

int main(int argc, char*argv[])
{

    Py_Initialize();
    PyRun_SimpleString("print('Hello C++ from Python')");
    PyObject * obj = Py_BuildValue("s", "driver_1d.py"); // load objects in variable
    FILE * fp = _Py_fopen_obj(obj, "r+");
    if(fp != NULL)
    {
        PyRun_SimpleFile(fp, "driver_1d.py");
    }
    Py_Finalize();

    return 0;
}