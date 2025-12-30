#include <AMReX_REAL.H>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *py_thermal_properties;
PyObject *ac_driver;
PyObject *y_queries;

inline
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

inline
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

#include <cmath>

struct KrigingModel {
    int n_samples;
    amrex::Real* X_norma;
    amrex::Real* alpha; // gamma
    amrex::Real* theta;
    amrex::Real beta;
    amrex::Real X_offset;
    amrex::Real X_scale;
    amrex::Real y_mean;
    amrex::Real y_std;

    AMREX_GPU_HOST_DEVICE
    amrex::Real predict(amrex::Real x) const {
        amrex::Real x_norm = (x - X_offset) / X_scale;
        amrex::Real y_norm = beta;
        for (int i = 0; i < n_samples; ++i) {
            amrex::Real dist_sq = (x_norm - X_norma[i]) * (x_norm - X_norma[i]);
            amrex::Real corr = std::exp(-theta[0] * dist_sq);
            y_norm += alpha[i] * corr;
        }
        return y_norm * y_std + y_mean;
    }
};

AMREX_GPU_DEVICE_MANAGED KrigingModel* d_kriging_model = nullptr;

// Helper to copy numpy array to device memory
amrex::Real* copy_numpy_to_device(PyObject* dict, const char* key, int& size) {
    PyObject* arr = PyDict_GetItemString(dict, key);
    if (!arr) amrex::Abort(std::string("Missing key: ") + key);
    
    PyArrayObject* np_arr = (PyArrayObject*)arr;
    size = PyArray_SIZE(np_arr);
    double* data = (double*)PyArray_DATA(np_arr);
    
    amrex::Real* d_ptr = (amrex::Real*)amrex::The_Arena()->alloc(size * sizeof(amrex::Real));
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, data, data + size, d_ptr);
    return d_ptr;
}

amrex::Real get_scalar_from_dict(PyObject* dict, const char* key) {
    PyObject* item = PyDict_GetItemString(dict, key);
    if (!item) amrex::Abort(std::string("Missing key: ") + key);
    if (PyArray_Check(item)) {
         return (amrex::Real)(*(double*)PyArray_DATA((PyArrayObject*)item));
    }
    return (amrex::Real)PyFloat_AsDouble(item);
}

void update_gpu_model_from_python(PyObject* py_module, PyObject* ac_driver) {
    // 1. Call get_kriging_params
    PyObject* params = PyObject_CallMethod(py_module, "get_kriging_params", "O", ac_driver);
    if (!params) {
        PyErr_Print();
        amrex::Abort("Failed to get kriging params");
    }

    // 2. Allocate/Update Host Struct
    KrigingModel h_model;
    int size;
    
    h_model.X_norma = copy_numpy_to_device(params, "X_norma", size);
    h_model.n_samples = size;
    
    h_model.alpha = copy_numpy_to_device(params, "gamma", size);
    h_model.theta = copy_numpy_to_device(params, "theta", size);
    
    h_model.beta = get_scalar_from_dict(params, "beta");
    h_model.X_offset = get_scalar_from_dict(params, "X_offset");
    h_model.X_scale = get_scalar_from_dict(params, "X_scale");
    h_model.y_mean = get_scalar_from_dict(params, "y_mean");
    h_model.y_std = get_scalar_from_dict(params, "y_std");
    
    // 3. Copy Struct to Device
    if (d_kriging_model == nullptr) {
        d_kriging_model = (KrigingModel*)amrex::The_Arena()->alloc(sizeof(KrigingModel));
    }
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, &h_model, &h_model + 1, d_kriging_model);
    
    Py_DECREF(params);
    amrex::Print() << "GPU Kriging Model Updated. n_samples=" << h_model.n_samples << std::endl;
}

AMREX_GPU_HOST_DEVICE inline
amrex::Real get_thermal_conductivity(amrex::Real temperature)
{
#ifndef AMREX_USE_GPU
  // CPU Fallback to Python if model not ready or explicitly on CPU
  if (d_kriging_model == nullptr) {
      PyObject *x_queries = PyList_New(1);
      PyList_SetItem(x_queries, 0, PyFloat_FromDouble(temperature));

      PyObject *y_queries = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_queries, "absolute_variance", 1.e-9);
      if (y_queries == NULL) {
        amrex::Abort"query failed");
        return 16.0; // Fallback
      }
      amrex::Real kappa = get_double_from_entry(y_queries,0,0);
      Py_DECREF(y_queries);
      Py_DECREF(x_queries);
      return std::max(1e-10, kappa);
  }
#endif

  // GPU / Fast Path
  if (d_kriging_model) {
      return std::max(1e-10, d_kriging_model->predict(temperature));
  }
  
  // Fallback if model not initialized
  return 16.0 + 0.01 * (temperature - 300.0);
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
