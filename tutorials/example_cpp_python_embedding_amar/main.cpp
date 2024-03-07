//

#include <python.h>
#include <iostream>

int main(int argc, char*argv[])
{

    Py_Initialize();    
    PyRun_SimpleString("print('Hello C++ from Python')");

    // Initialize the DataSet. In continuum simulation, the dataset the flow field. Temperature, Pressure, Composition (mass fraction for the species). Load the dataset and some initial (random or tabulated) data points. The dataset will keep track of which microscale simulations have been performed.


    PyRun_SimpleString("import sys");
    PyRun_SimpleString(R"(sys.path.insert(0, r'C:\Users\JanelleDomantay\Documents\GitHub\AdaptiveComputing'))");
    PyRun_SimpleString("from ac_common import *");
    //import module    

    PyObject* myModule = PyImport_ImportModule("functions");
    
    if (myModule == nullptr){
      std::cerr << "Failed to import module.\n";
      return 1;
    }


    //initalize dataset
    PyObject* init_dataset = PyObject_GetAttrString(myModule, (char*)"init_dataset");
    PyObject* my_dataset = PyObject_CallObject(init_dataset, nullptr);

    // Initialize the surrogate model. The surrogate model allows us to interpolate between those simulations.
    PyObject* init_surrogate = PyObject_GetAttrString(myModule, (char*)"init_surrogate");
    PyObject* args = PyTuple_Pack(1, my_dataset);
    PyObject* my_surrogate = PyObject_CallObject(init_surrogate, args); //pass in dataset to surrogate initalization   

    // Run the continuum simulation
    int N_iter = 5;
    for (int n = 0; n < N_iter; n++){
      // The real code will do some complex code to determine the T,P,x0,x1 values where the surrogate model needs to be queried
      int T = 20.0+n/2.0;
      int P = 0.5+n/10.0;
      int x0 = n/N_iter;
      int x1 = 1.0-x0;

      // Query the surrogate model. If the variances is too high, run a simulation, otherwise, interpolate the surrogate model.

      int x_query[4] = {T, P, x0, x1};
      double threshold_std_mean = 0.5;

      PyObject* query_surrogate = PyObject_GetAttrString(myModule, (char*)"query_surrogate");
      args = PyTuple_Pack(2, my_dataset, my_surrogate);
      PyObject* y_query = PyObject_CallObject(query_surrogate, args);
      // This is the python code we want:
      // x_query = [T, P, x0, x1]
      // thereshold_std_mean = 0.5
      // y_query, y_query_var = my_dataset.query(my_surrogate,x_query,threshold_std_mean=threshold_std_mean)
      // Not sure how to do this. Maybe we just need to add input arguments to this: PyObject * y_query = PyObject_CallMethod(my_dataset, "query", nullptr);

      // Next step: once we get this working, we can implement func_4d in cpp (since KMC is implemented in cpp) and we can figure out how we want to have AC tell this file to run the KMC simulation and report the result back to AC.
    }

    Py_Finalize();

    return 0;
}
