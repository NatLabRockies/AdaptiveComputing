#include <ac.H>
#include <cmath>

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_BLassert.H>
#include <AMReX_SPACE.H>
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>


using namespace amrex;

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

