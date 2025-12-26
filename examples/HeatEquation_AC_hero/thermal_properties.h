#include <AMReX_REAL.H>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *py_thermal_properties;
PyObject *ac_driver;
PyObject *y_queries;

double get_double_from_entry(PyObject* py_array, int row, int col) {
    // Cast PyObject* to PyArrayObject*
    PyArrayObject* in_array = (PyArrayObject*)py_array;

    // Ensure array is of type NPY_DOUBLE
    if (PyArray_TYPE(in_array) != NPY_DOUBLE) {
        // Handle error: array is not doubles
      std::cerr << "Entries are not doubles" << '\n';
        return 0.0; 
    }

    // Get a pointer to the raw data
    double* data_ptr = (double*)PyArray_DATA(in_array);

    // Calculate the index offset for a 2D array (assuming row-major C style)
    npy_intp* dims = PyArray_DIMS(in_array);
    int rows = dims[0];
    int cols = dims[1];

    // Check bounds
    if (row >= rows || col >= cols || row < 0 || col < 0) {
      std::cerr << "Indices OOB" << '\n';
        // Handle error: index out of bounds
        return 0.0;
    }

    // Access the value
    double value = data_ptr[row * cols + col];

    return value;
}

AMREX_GPU_HOST_DEVICE
amrex::Real get_thermal_conductivity(amrex::Real temperature)
{
  PyObject *x_queries = PyList_New(1);
  PyList_SetItem(x_queries, 0, Py_BuildValue("[d]", temperature));

  // Call ac_driver.query, returns a numpy ndarray type object
  PyObject *y_queries = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_queries, "absolute_variance", 0.1);
  if (!y_queries) { PyErr_Print(); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(py_thermal_properties); std::cout << "FATAL Error" << '\n'; return -1; }
    
  amrex::Real kappa = get_double_from_entry(y_queries,0,0);

  //std::cout << "T-Dependent Alpha calc called for T="<< temperature << '\n';
  // Ensure kappa is physically meaningful (non-negative)
  Py_DECREF(y_queries);
  Py_DECREF(x_queries);
  return std::max(1e-10, kappa);
}

amrex::Real get_SpecificHeatCapacity(amrex::Real temperature) {
    amrex::Real Cp = 500 + 0.1 * (temperature-300);
    return std::max(1e-10, Cp);
}

amrex::Real get_density(amrex::Real temperature) {
    amrex::Real rho = 7910 - 0.4 * (temperature-300);
    return std::max(1e-10, rho);
}
