//
#include <python.h>
#include <iostream>
#include "func_4d.h"

PyObject* initializeIntArray(int* array, int length) {
    // Create a Python list from the C integer array
    PyObject* pyList = PyList_New(length);
    if (pyList == NULL) {
        PyErr_Print();
        return NULL;
    }

    for (int i = 0; i < length; ++i) {
        // Create a Python integer object for each element of the array
        PyObject* pyInt = PyLong_FromLong(array[i]);
        if (pyInt == NULL) {
            PyErr_Print();
            Py_DECREF(pyList);
            return NULL;
        }
        // Set the Python integer object in the list
        PyList_SET_ITEM(pyList, i, pyInt);
    }

    // Return the Python list object
    return pyList;
}

int main(int argc, char*argv[])
{

    Py_Initialize();    

    PyObject* myModule = PyImport_ImportModule("func");
    PyObject * obj = Py_BuildValue("s", "func.py"); // load objects in variable
    FILE * fp = _Py_fopen_obj(obj, "r+");
    if(fp != NULL)
    {
        PyRun_SimpleFile(fp, "func.py");
    }


    if (myModule == nullptr){
      std::cerr << "Failed to import module.\n";
      Py_DECREF(myModule);
      Py_DECREF(obj);
      Py_Finalize();
      return 1;
    }


    //initalize dataset
    PyObject* init_dataset = PyObject_GetAttrString(myModule, (char*)"init_dataset");
    PyObject* my_dataset = PyObject_CallObject(init_dataset, nullptr);
    //PyObject* my_dataset = PyObject_CallMethod(myModule, "init_dataset", nullptr);

    // Initialize the surrogate model. The surrogate model allows us to interpolate between those simulations.
    //PyObject* init_surrogate = PyObject_GetAttrString(myModule, (char*)"init_surrogate");
    //PyObject* args = PyTuple_Pack(1, my_dataset);
    //PyObject* my_surrogate = PyObject_CallObject(init_surrogate, args); //pass in dataset to surrogate initalization   
    PyObject* my_surrogate = PyObject_CallMethod(myModule, "init_surrogate", "O", my_dataset);

    // Run the continuum simulation
    int N_iter = 5;
    for (int n = 0; n < N_iter; n++){
      // The real code will do some complex code to determine the T,P,x0,x1 values where the surrogate model needs to be queried
      int T = 20.0+n/2.0;
      int P = 0.5+n/10.0;
      int x0 = n/N_iter;
      int x1 = 1.0-x0;

      // Query the surrogate model. If the variances is too high, run a simulation, otherwise, interpolate the surrogate model.
      
      double threshold_std_mean = 0.5;

      int cpp_x_query[] = {T, P, x0, x1};
      int x[4] = {T, P, x0, x1};
      int length = sizeof(cpp_x_query) / sizeof(cpp_x_query[0]);

      PyObject* x_queries = initializeIntArray(cpp_x_query, length);
      if (x_queries == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
      }

    PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean);
    
      //PyObject* y_query = PyObject_CallMethod(my_dataset, "query_cpp", "OOsd", my_surrogate, x_queries, "threshold_std_mean", threshold_std_mean);
    if (y_query == NULL){
      //query returns mean 
      //if y_query == NULL,           
          //double y_val = func_4d(cpp_x_query); //call func4d.cpp
          //PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean);// call add_xnum_sample
    }
      
      //print y_query
      
      // This is the python code we want:
      // x_query = [T, P, x0, x1]
      // thereshold_std_mean = 0.5
      // y_query, y_query_var = my_dataset.query(my_surrogate,x_query,threshold_std_mean=threshold_std_mean)
      // Not sure how to do this. Maybe we just need to add input arguments to this: PyObject * y_query = PyObject_CallMethod(my_dataset, "query", nullptr);

      // Next step: once we get this working, we can implement func_4d in cpp (since KMC is implemented in cpp) and we can figure out how we want to have AC tell this file to run the KMC simulation and report the result back to AC.
      //Dereference
      Py_DECREF(x_queries);
      Py_DECREF(y_query);
    }

    Py_DECREF(myModule);
    Py_DECREF(obj);
    Py_DECREF(my_dataset);
    Py_DECREF(my_surrogate);
    //Py_DECREF(args);

    Py_Finalize();

    return 0;
}

