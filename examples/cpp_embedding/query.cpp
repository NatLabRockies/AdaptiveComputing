#include <Python.h>
#include <iostream>

// This code implements the __main__ function of py_query.py but in C++
// using Python embedding.
// First, a surrogate model is initialized with a few random samples.
// Then, it queries the surrogate at some locations (x_queries) and compares
// the resulting variances with a constant variance threshold value.
// For each location where the threshold is exceeded, a simulation is run and
// the surrogate is retrained using the simulation result.
// Otherwise, it trusts the surrogate. The query results (y_queries) are printed along
// with the x_data and y_data in the surrogate's training dataset.

int main() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Import sys module
    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) { PyErr_Print(); return -1; }

    // Append the working directory to sys.path
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyObject *cwd = PyUnicode_FromString(".");
    PyList_Append(path, cwd);
    Py_DECREF(cwd);
    Py_DECREF(path);
    Py_DECREF(sys);

    // Import py_query module
    PyObject *py_query = PyImport_ImportModule("py_query");
    if (!py_query) { PyErr_Print(); return -1; }

    // Call initialize_driver()
    PyObject *ac_driver = PyObject_CallMethod(py_query, "initialize_driver", nullptr);
    if (!ac_driver) { PyErr_Print(); Py_DECREF(py_query); return -1; }

    // Call print_data(ac_driver)
    PyObject *temp = PyObject_CallMethod(py_query, "print_data", "O", ac_driver);
    if (!temp) { PyErr_Print(); Py_DECREF(ac_driver); Py_DECREF(py_query); return -1; }
    Py_DECREF(temp);

    // Create list of x_queries = [[1],[3],[6],[2.8],[5]]
    PyObject *x_queries = PyList_New(5);
    if (!x_queries) { PyErr_Print(); Py_DECREF(ac_driver); Py_DECREF(py_query); return -1; }

    PyList_SetItem(x_queries, 0, Py_BuildValue("[d]", 1.0));
    PyList_SetItem(x_queries, 1, Py_BuildValue("[d]", 3.0));
    PyList_SetItem(x_queries, 2, Py_BuildValue("[d]", 6.0));
    PyList_SetItem(x_queries, 3, Py_BuildValue("[d]", 2.8));
    PyList_SetItem(x_queries, 4, Py_BuildValue("[d]", 5.0));

    // Print the x_queries
    PyObject *repr = PyObject_Repr(x_queries);
    if (repr) {
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << "x_queries = " << str << std::endl;
        Py_DECREF(repr);
    }

    // Call ac_driver.query
    PyObject *y_queries = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_queries, "absolute_variance", 0.1);
    if (!y_queries) { PyErr_Print(); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(py_query); return -1; }

    // Print y_queries
    repr = PyObject_Repr(y_queries);
    if (repr) {
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << "y_queries = " << str << std::endl;
        Py_DECREF(repr);
    }

    // Call print_data(ac_driver)
    temp = PyObject_CallMethod(py_query, "print_data", "O", ac_driver);
    if (!temp) { PyErr_Print(); Py_DECREF(y_queries); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(py_query); return -1; }
    Py_DECREF(temp);

    // Call ac_driver.query again (this time it should not run any new simulations since we are querying at the same locations as before)
    PyObject *y_queries2 = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_queries, "absolute_variance", 0.1);
    if (!y_queries2) { PyErr_Print(); Py_DECREF(y_queries); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(py_query); return -1; }

    // Print y_queries
    repr = PyObject_Repr(y_queries2);
    if (repr) {
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << "y_queries (second call) = " << str << std::endl;
        Py_DECREF(repr);
    }

    // Call print_data(ac_driver)
    temp = PyObject_CallMethod(py_query, "print_data", "O", ac_driver);
    if (!temp) { PyErr_Print(); Py_DECREF(y_queries2); Py_DECREF(y_queries); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(py_query); return -1; }
    Py_DECREF(temp);

    // Cleanup
    Py_DECREF(y_queries2);
    Py_DECREF(y_queries);
    Py_DECREF(x_queries);
    Py_DECREF(ac_driver);
    Py_DECREF(py_query);

    // Finalize the Python interpreter
    Py_Finalize();

    return 0;
}
