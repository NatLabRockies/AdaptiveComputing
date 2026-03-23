#include <Python.h>
#include <iostream>

// This code implements the __main__ function of controller_offline_inference.py but in C++
// using Python embedding.
// First, a surrogate model is loaded from a file.
// Then, it is evaulated at several locations (x_queries).

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

    // Import controller_offline_inference.py module
    PyObject *controller_offline_inference = PyImport_ImportModule("controller_offline_inference");
    if (!controller_offline_inference) { PyErr_Print(); return -1; }

    // Call initialize_driver()
    PyObject *ac_driver = PyObject_CallMethod(controller_offline_inference, "initialize_driver", nullptr);
    if (!ac_driver) { PyErr_Print(); Py_DECREF(controller_offline_inference); return -1; }

    // Call print_data(ac_driver)
    PyObject *temp = PyObject_CallMethod(controller_offline_inference, "print_data", "O", ac_driver);
    if (!temp) { PyErr_Print(); Py_DECREF(ac_driver); Py_DECREF(controller_offline_inference); return -1; }
    Py_DECREF(temp);

    // Create list of x_queries = [[0.85],[0.9],[1.1],[1.5],[2.0]]
    PyObject *x_queries = PyList_New(5);
    if (!x_queries) { PyErr_Print(); Py_DECREF(ac_driver); Py_DECREF(controller_offline_inference); return -1; }

    PyList_SetItem(x_queries, 0, Py_BuildValue("[d]", 0.85));
    PyList_SetItem(x_queries, 1, Py_BuildValue("[d]", 0.9));
    PyList_SetItem(x_queries, 2, Py_BuildValue("[d]", 1.1));
    PyList_SetItem(x_queries, 3, Py_BuildValue("[d]", 1.5));
    PyList_SetItem(x_queries, 4, Py_BuildValue("[d]", 2.0));

    // Print the x_queries
    PyObject *repr = PyObject_Repr(x_queries);
    if (repr) {
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << "x_queries = " << str << std::endl;
        Py_DECREF(repr);
    }

    // Call ac_driver.surrogate.predict_values(x_queries)
    PyObject *y_queries = PyObject_CallMethod(controller_offline_inference, "predict_values", "O,O", ac_driver, x_queries);
    if (!y_queries) { PyErr_Print(); Py_DECREF(x_queries); Py_DECREF(ac_driver); Py_DECREF(controller_offline_inference); return -1; }

    // Print y_queries
    repr = PyObject_Repr(y_queries);
    if (repr) {
        const char *str = PyUnicode_AsUTF8(repr);
        std::cout << "y_queries = " << str << std::endl;
        Py_DECREF(repr);
    }

    // Cleanup
    Py_DECREF(y_queries);
    Py_DECREF(x_queries);
    Py_DECREF(ac_driver);
    Py_DECREF(controller_offline_inference);

    // Finalize the Python interpreter
    Py_Finalize();

    return 0;
}
