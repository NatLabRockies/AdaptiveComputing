#include <AMReX_REAL.H>
#include <iostream>
#include <Python.h>

PyObject *py_thermal_properties;
PyObject *ac_driver;
PyObject *y_queries;

double get_double_from_list(PyObject* float_list, Py_ssize_t index) {
    PyObject* item;
    double value;
    // 1. Retrieve the PyObject* item from the list
    item = PyList_GetItem(float_list, index); // Returns a borrowed reference
    if (item == NULL) {
        // PyList_GetItem returns NULL if the index is out of bounds, an exception is set.
        return -1.0; // Or handle the error as appropriate for your application
    }
    // 2. Check the type (optional but recommended for robustness)
    if (PyFloat_Check(item)) {
        // 3. Convert to double
        value = PyFloat_AsDouble(item);
    } else if (PyLong_Check(item)) {
        // Optionally handle Python integers, converting them to double
        value = (double)PyLong_AsLong(item);
    } else {
        // Set a Python TypeError if the object is not a number
        PyErr_SetString(PyExc_TypeError, "List item is not a float or integer");
        return -1.0; // Indicate failure
    }
    // 4. Check for errors during conversion
    if (PyErr_Occurred()) {
        return -1.0; // Indicate failure
    }
    return value;
}

AMREX_GPU_HOST_DEVICE
amrex::Real get_thermal_conductivity(amrex::Real temperature) {

    std::cout << "test1" << std::endl;

    PyObject *x_queries = PyList_New(1);
    if (!x_queries) { PyErr_Print(); Py_DECREF(ac_driver); Py_DECREF(py_thermal_properties); return -1; }

    std::cout << "test2" << std::endl;

    PyList_SetItem(x_queries, 0, Py_BuildValue("[d]", temperature));    
    // Print the x_queries
    PyObject *repr = PyObject_Repr(x_queries);
    if (repr) {
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << "x_queries = " << str << std::endl;
        Py_DECREF(repr);
    }
    std::cout << "test3" << std::endl;
    // Call ac_driver.query
    PyObject *y_queries = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_queries, "absolute_variance", 0.1);
    if (!y_queries) { PyErr_Print(); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(py_thermal_properties); std::cout << "FATAL Error" << std::endl; return -1; }
    std::cout << "test4" << std::endl;
    
    //amrex::Real kappa=get_double_from_list(y_queries,0);
    amrex::Real kappa = PyFloat_AsDouble(y_queries);
    std::cout << "kappa=" << kappa << std::endl; 
    // Simple linear model for temperature-dependent diffusivity
    //amrex::Real kappa = 16 + 0.01 * (temperature-400);

    //std::cout << "T-Dependent Alpha calc called for T="<< temperature << std::endl;    
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
