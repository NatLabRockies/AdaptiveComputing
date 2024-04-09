//
#include <python.h>
#include <iostream>
#include "func_4d.h"
#include <ctime>

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

    //train on dataset.train_on_unmasked_data(self)

    // Run the continuum simulation
    int N_iter = 3; //for running sanity test
    for (int n = 0; n < N_iter; n++){
      // The real code will do some complex code to determine the T,P,x0,x1 values where the surrogate model needs to be queried
      float T = 20.0+n/2.0;
      float P = 0.5+n/10.0;
      float x0 = n/N_iter;
      float x1 = 1.0-x0;

      double threshold_std_mean = 0.5;

      //sanity tests
      if (n == 0){ // query pt in dataset, expect 0 variance, should not re evaluate
        PyRun_SimpleString("print('Running Test 1, low variance, no re-evaluation')");
        T = 96.1335675;
        P = 7.20324493;
        x0 = 0.414038694;
        x1 = 0.268521950;
      }
      else if (n > 0){ //query a point not in the training set with high threshold, should not re evaluate
        if (n == 1)
        {
          PyRun_SimpleString("print('Running Test 2, high threshold, high variance no re-evaluation')");
          threshold_std_mean = 100;
        }
        else if (n == 2){
          PyRun_SimpleString("print('Running Test 3, low threshold, high variance re-evaluation')");
          threshold_std_mean = 0.0001;
        }
      }      
      // Query the surrogate model. If the variances is too high, run a simulation, otherwise, interpolate the surrogate model.
      
      

      float cpp_x_query[] = {T, P, x0, x1};
      float x[4] = {T, P, x0, x1};
      int length = sizeof(cpp_x_query) / sizeof(cpp_x_query[0]);

      PyObject* x_queries = initializeFloatArray(cpp_x_query, length);
      if (x_queries == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
      }

      //PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean); //query dataset
      PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, time_ratio, computer_budget_ratio);//dynamic query to dataset
           
      

      if (y_query == Py_None){         
        int start = clock();
        double y_val = func_4d(cpp_x_query); //query original function via cpp call
        int stop = clock(); //timing code here
        PyObject_CallMethod(myModule, "add_xnum_sample", "OdOd", my_dataset, -1, x_queries, y_val); //PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean);// call add_xnum_sample
      }
      // else query returns mean, no need to sample
      
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

