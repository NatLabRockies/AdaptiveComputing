#include <AMReX_REAL.H>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *py_thermal_properties;
PyObject *ac_driver;
PyObject *y_queries;

AMREX_GPU_HOST_DEVICE inline
double get_double_from_entry(PyObject* py_array, int row, int col) {
    // Cast PyObject* to PyArrayObject*
    PyArrayObject* in_array = (PyArrayObject*)py_array;

    // Ensure array is of type NPY_DOUBLE
    if (PyArray_TYPE(in_array) != NPY_DOUBLE) {
      amrex::Abort("Entries are not doubles");
    }

    // Get a pointer to the raw data
    double* data_ptr = (double*)PyArray_DATA(in_array);

    // Calculate the index offset for a 2D array (assuming row-major C style)
    npy_intp* dims = PyArray_DIMS(in_array);
    int ndim = PyArray_NDIM(in_array);
    AMREX_ALWAYS_ASSERT(ndim==1 || ndim==2);
    int rows = dims[0];
    int cols = ndim==2 ? dims[1] : 1;

    // Check bounds
    if (row >= rows || col >= cols || row < 0 || col < 0) {
      amrex::Abort("Indices OOB");
    }

    // Access the value
    double value = data_ptr[row * cols + col];

    return value;
}

AMREX_GPU_HOST_DEVICE inline
void set_double_at_entry(PyObject* py_array, int row, int col, double value) {
    // Cast PyObject* to PyArrayObject*
    PyArrayObject* in_array = (PyArrayObject*)py_array;

    // Ensure array is of type NPY_DOUBLE
    if (PyArray_TYPE(in_array) != NPY_DOUBLE) {
      amrex::Abort("Entries are not doubles");
    }

    // Get a pointer to the raw data
    double* data_ptr = (double*)PyArray_DATA(in_array);

    // Calculate the index offset for a 2D array (assuming row-major C style)
    npy_intp* dims = PyArray_DIMS(in_array);
    int ndim = PyArray_NDIM(in_array);
    AMREX_ALWAYS_ASSERT(ndim==1 || ndim==2);
    int rows = dims[0];
    int cols = ndim==2 ? dims[1] : 1;

    // Check bounds
    if (row >= rows || col >= cols || row < 0 || col < 0) {
      amrex::Abort("Indices OOB");
    }

    // Access the value
    data_ptr[row * cols + col] = value;
}

AMREX_GPU_HOST_DEVICE inline
amrex::Real get_thermal_conductivity(amrex::Real temperature)
{
  PyObject *x_queries = PyList_New(1);
  PyList_SetItem(x_queries, 0, PyFloat_FromDouble(temperature));

  // Call ac_driver.query, returns a numpy ndarray type object
  PyObject *y_queries = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_queries, "absolute_variance", 1.e-9);
  if (y_queries == NULL) {
    amrex::Print() << "query failed" << std::endl;
    PyErr_Print();
  }
  amrex::Real kappa = get_double_from_entry(y_queries,0,0);

  //std::cout << "T-Dependent Alpha calc called for T="<< temperature << '\n';
  // Ensure kappa is physically meaningful (non-negative)
  Py_DECREF(y_queries);
  Py_DECREF(x_queries);
  return std::max(1e-10, kappa);
}

AMREX_GPU_HOST_DEVICE inline
amrex::Real get_SpecificHeatCapacity(amrex::Real temperature) {
    amrex::Real Cp = 500 + 0.1 * (temperature-300);
    return std::max(1e-10, Cp);
}

AMREX_GPU_HOST_DEVICE inline
amrex::Real get_density(amrex::Real temperature) {
    amrex::Real rho = 7910 - 0.4 * (temperature-300);
    return std::max(1e-10, rho);
}
