//
#include <python.h>
#include <iostream>
#include "func_4d.h"
#include <ctime>
#include <math.h>


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
         
    double wall_clock_end = double(clock()) + 100000; //specifies total amount of time to run simulation
    double cpu_elapsed = 0;
    double cpu_budget = 50000; //specifies amount of computing hours budgeted
    double hrs_per_sim = 500; //specifies projected time it takes for one simulation



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
    //   

    // Initialize the surrogate model. The surrogate model allows us to interpolate between those simulations.

    PyObject* my_surrogate = PyObject_CallMethod(myModule, "init_surrogate", "O", my_dataset);
  
    // Run the continuum simulation
    int N_iter = 100; 
    int thres_iter = 10;
    int boundary_pts = 64; 
    int count_HF = 0;
    int count_LF = 0;
    int n = 0;

    
    //for (int n = 0; n < N_iter; n++){    
    while (double(clock()) < wall_clock_end)
    {      
      n++;
      // The real code will do some complex code to determine the T,P,x0,x1 values where the surrogate model needs to be queried
      //generate 8^2 candidates
      float T [boundary_pts];
      float P [boundary_pts];
      float x0 [boundary_pts];
      float x1 [boundary_pts];
      double t_0 = 0.1;
      double t_1 = 1.1*t_0;
      double r_0 = 0;
      double threshold_std_mean = 0.05;

      for (int m = 0; m < thres_iter; m++){// loop over variance threshold choices   t                
        int num_simulations = 0;
        
        for (int i = 0; i < boundary_pts; i++){
          if (m == 0){
            T[i] = 20.0 + float(float(n) + float(i)/float(boundary_pts)) / 2.0;//20.0+float(n + float(i/(boundary_pts)))/2.0;
            P[i] = 0.5 + float(float(n) + float(i)/float(boundary_pts)) / 10.0;//0.5+float(n + float(i/(boundary_pts)) )/10.0;
            x0[i] = float(float(n) + float(i)/float(boundary_pts)) / float(N_iter);//float(n + float(i/(boundary_pts)) )/N_iter;
            x1[i] = 1.0-x0[i];
          }

          float cpp_x_query[] = {T[i], P[i], x0[i], x1[i]};
          float x[4] = {T[i], P[i], x0[i], x1[i]};
          int length = sizeof(cpp_x_query) / sizeof(cpp_x_query[0]);

          PyObject* x_queries = initializeFloatArray(cpp_x_query, length); //used to convert c++ float array to PyObject
          if (x_queries == NULL) {
            PyErr_Print();
            Py_Finalize();
            return 1;
          }

          PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean); //query dataset
          if (y_query == Py_None){      
            num_simulations += 1;
          }

          if (m == thres_iter - 1) { //simulation on last iteration
            if (y_query == Py_None && cpu_elapsed < cpu_budget){         
              count_HF += 1;
              int start = clock();
              double y_val = func_4d(cpp_x_query); //query original function via cpp call
              int stop = clock(); //timing code here
              cpu_elapsed += stop - start;
              PyObject_CallMethod(myModule, "add_xnum_sample", "OdOd", my_dataset, -1, x_queries, y_val); //PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean);// call add_xnum_sample
            }
            else{       // else query returns mean, no need to sample
              count_LF += 1;
            }
            double curr_time = double(clock());
            PyObject_CallMethod(myModule, "write_output", "idddii", (n - 1) * boundary_pts + i,  cpu_elapsed/ cpu_budget, curr_time / wall_clock_end, threshold_std_mean,  count_HF, count_LF);
          }        
          //Dereference
          Py_DECREF(x_queries);
          Py_DECREF(y_query);
        }
        double curr_time = double(clock());
        double time_ratio = curr_time/ wall_clock_end;
        double computer_budget_ratio = (cpu_elapsed + num_simulations * hrs_per_sim)/ cpu_budget;
        //update variance threshold based on num_simulations      

        if (m < thres_iter - 1){ //update threshold values
          //secant method
          double r_1 = computer_budget_ratio - time_ratio;
          double t_2 = t_1 - r_1 * (t_1 - t_0) / float(r_1 - r_0);
          r_0 = r_1;
          t_0 = t_1;
          t_1 = t_2;
          threshold_std_mean = t_2;
          /*
            if (computer_budget_ratio < time_ratio){
              threshold_std_mean -= t_2;//0.01 * threshold_std_mean; //threshold_std_mean * 0.5; //(time_ratio - computer_budget_ratio);
            }
            else{
              threshold_std_mean += t_2;//0.15 * threshold_std_mean; //threshold_std_mean * 0.5; //(computer_budget_ratio - time_ratio);
            }
          */
          if (threshold_std_mean < 0){
            threshold_std_mean = 0.00001;
          }
          if (m == thres_iter - 2){
            if (computer_budget_ratio > 1){
              threshold_std_mean = INFINITY;
            }
          }
        }                              
      }    
      //check for convergence print out ratios , print out ratios and threshold

      // Query the surrogate model. If the variances is too high, run a simulation, otherwise, interpolate the surrogate model.            
      //float cpp_x_query[] = {T, P, x0, x1};
      //float x[4] = {T, P, x0, x1};
      //int length = sizeof(cpp_x_query) / sizeof(cpp_x_query[0]);

      //PyObject* x_queries = initializeFloatArray(cpp_x_query, length);
      //if (x_queries == NULL) {
      //  PyErr_Print();
      //  Py_Finalize();
      //  return 1;
      //}

      //PyObject* y_query = PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean); //query dataset
      //PyObject* y_query = PyObject_CallMethod(myModule, "dynamic_if_query", "OOOdd", my_dataset, my_surrogate, x_queries, time_ratio, computer_budget_ratio);//dynamic query to dataset
                 
    }

    PyObject_CallMethod(myModule, "print_stmt", "iid", count_LF, count_HF, cpu_elapsed);

    Py_DECREF(myModule);
    Py_DECREF(obj);
    Py_DECREF(my_dataset);
    Py_DECREF(my_surrogate);
    //Py_DECREF(args);

    Py_Finalize();

    return 0;
}

