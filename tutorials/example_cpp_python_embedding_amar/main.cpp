//
#include <Python.h>
#include <iostream>
#include "func_4d.h"
#include <ctime>
#include <math.h>
#include <iostream>

PyObject* initializeFloatArray(float* array, int length) {
    // Create a Python list from the C integer array
    PyObject* pyList = PyList_New(length);
    if (pyList == NULL) {
        PyErr_Print();
        return NULL;
    }

    for (int i = 0; i < length; ++i) {
        // Create a Python integer object for each element of the array
        PyObject* pyFloat = PyFloat_FromDouble(array[i]);
        if (pyFloat == NULL) {
            PyErr_Print();
            Py_DECREF(pyList);
            return NULL;
        }
        // Set the Python integer object in the list
        PyList_SET_ITEM(pyList, i, pyFloat);
    }

    // Return the Python list object
    return pyList;
}

int main(int argc, char*argv[])
{

    Py_Initialize();    
         
    double cpu_elapsed = 0;

    PyObject* sys = PyImport_ImportModule("sys");
    PyObject* path = PyObject_GetAttrString(sys, "path");
    Py_DECREF(sys);
    PyObject* cur_dir = PyUnicode_FromString("");
    PyList_Append(path, cur_dir);
    Py_DECREF(path);
    Py_DECREF(cur_dir);
    PyObject* myModule = PyImport_ImportModule("func");
    PyObject * obj = Py_BuildValue("s", "func.py"); // load objects in variable
    FILE * fp = _Py_fopen_obj(obj, "r+");
    Py_DECREF(obj);
    if(fp != NULL)
    {
        PyRun_SimpleFile(fp, "func.py");
	fclose(fp);
    } else
      assert(0);
    

    if (myModule == nullptr){
      std::cerr << "Failed to import module.\n";
      Py_DECREF(myModule);
      Py_Finalize();
      return 1;
    }


    //initalize dataset
    PyObject* init_dataset = PyObject_GetAttrString(myModule, (char*)"init_dataset");
    PyObject* my_dataset = PyObject_CallObject(init_dataset, nullptr);
    Py_DECREF(init_dataset);

    // Initialize the surrogate model. The surrogate model allows us to interpolate between simulations.
    PyObject* my_surrogate = PyObject_CallMethod(myModule, "init_surrogate", "O", my_dataset);
  
    // Run the continuum simulation
    int N_iter = 2; // number of continuum time steps
    int N_boundary_pts = 2; // number of grid points
    int count_HF = 0;
    int count_LF = 0;
    int n = 0;

    
    for (int n = 0; n < N_iter; n++){    
      // The real code will do some complex code to determine the T,P,x0,x1 values where the surrogate model needs to be queried
      //generate 8^2 candidates
      float T [N_boundary_pts];
      float P [N_boundary_pts];
      float x0 [N_boundary_pts];
      float x1 [N_boundary_pts];
      double threshold = 0.1;

      for (int i = 0; i < N_boundary_pts; i++){
	T[i] = 20.0 + float(float(n) + float(i)/float(N_boundary_pts)) / 2.0;
	P[i] = 0.5 + float(float(n) + float(i)/float(N_boundary_pts)) / 10.0;
	x0[i] = float(float(n) + float(i)/float(N_boundary_pts)) / float(N_iter);
	x1[i] = 1.0-x0[i];
	
	float cpp_x_query[] = {T[i], P[i], x0[i], x1[i]};
	float x[4] = {T[i], P[i], x0[i], x1[i]};
	int length = sizeof(cpp_x_query) / sizeof(cpp_x_query[0]);

	PyObject* x_queries = initializeFloatArray(cpp_x_query, length); //used to convert c++ float array to PyObject
	if (x_queries == NULL) {
	  PyErr_Print();
	  Py_DECREF(myModule);
	  Py_Finalize();
	  return 1;
	}

	std::cout << "x_query = [" << cpp_x_query[0] << ", " << cpp_x_query[1] << ", " << cpp_x_query[2] << ", " << cpp_x_query[3] << "]" << std::endl;
	PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold); //query dataset
	if (y_query == NULL) {
	  PyErr_Print();
	  Py_DECREF(myModule);
	  Py_Finalize();
	  return 1;
	}

	if (y_query == Py_None){ // run a simulation
	  std::cout << "Variance threshold exceeded, running micro simulation." << std::endl;
	  count_HF += 1;
	  int start = clock();
	  double y_val = func_4d(cpp_x_query); //query original function via cpp call
	  int stop = clock(); //timing code here
	  cpu_elapsed += stop - start;
	  PyObject_CallMethod(myModule, "add_xnum_sample", "OiOdO", my_dataset, -1, x_queries, y_val, my_surrogate);
	}
	else{       // else query returns mean, no need to sample
	  count_LF += 1;
	}
	
	//Dereference
	Py_DECREF(x_queries);
	Py_DECREF(y_query);
      }
      double curr_time = double(clock());

    }

    std::cout << "Number of micro simulations run = " << count_HF << std::endl;
    std::cout << "Number of surrogates trusted = " << count_LF  << std::endl;
    std::cout << "CPU used on micro simulations = " << cpu_elapsed  << std::endl;

    Py_DECREF(myModule);

    Py_Finalize();

    return 0;
}

