Getting Started

Pre-requisites

    Python > 3.9
        module load anaconda3
        conda create --name acEnv
        conda activate acEnv

    SMT 1.3
        pip install smt=1.3

    numpy
        conda install numpy

    matplotlib
        conda install matplotlib

Usage

    Compilation Instructions: 
        Libraries for python must be specified via -L and -I flags
        i.e. g++ main.cpp -L /home/janelle9/.conda-envs/acEnv/lib -lpython3.XX -I /home/janelle9/.conda-envs/
        XX = Python version number

        Troubleshooting
            DSO Error - A DSO error is received during compilation
                Double check library path for libpython 3.XX.so
                Update LD_LIBRARY_PATH file to contain virtual environment library
                        
        Additional notes
            - the output file output only contains the cpp file and a python EMBEDDER, this means that any additional changes made to the referenced python files after compilation
            will be reflected when the embedder is called

Python Functions
    #include <Python.h> 
        - prerequisite for python embedding

    Py_Initialize(); 
        - used to initialize python compiler

    PyRun_SimpleString("print('Hello from Python')");
        - used to run a string of text in the Python interpreter

    PyObject* myModule = PyImport_ImportModule("func");  
        - allows you to import "func.py" as a module, you can call functions from func using PyObject_CallObject();
        
    PyObject * obj = Py_BuildValue("s", "func.py"); // load objects in variable
    FILE * fp = _Py_fopen_obj(obj, "r+");
    PyRun_SimpleFile(fp, "func.py");
        - can be used to run an entire file, "func.py" in its entirety, the equivalent of running python func.py

    PyObject* init_dataset = PyObject_GetAttrString(myModule, (char*) "init_dataset");
    PyObject* my_dataset = PyObject_CallObject(init_dataset, nullptr);
        - calls the function "init_dataset" from previously specified "func.py" module, arguments can be specified in the second parameter of PyObject_CallObject()
        - return values are stored in my_dataset

    PyObject* my_dataset = PyObject_CallMethod(myModule, "init_dataset", nullptr); 
    PyObject* my_surrogate = PyObject_CallMethod(myModule, "init_surrogate", "O", my_dataset);
        - an alternate method for calling a function from the func.py module, calls "init_surrogate" while passing in a PyObject "O" argument, "my_dataset"
        - This method can pass in parameters via PyObject_Call(myobject_method, args, keywords)
        - Note that for both the args and keywords argument, it is necessary to specify the datatype of the parameter
            PyObject_CallMethod(myModule, "if_query", "OOOd", my_dataset, my_surrogate, x_queries, threshold_std_mean); passes three PyObject files and a single double value as specified by "OOOd"
    

    Py_DECREF(myModule); 
        - dereferences PyObject 

    Py_Finalize();
        - used to close interpreter 




    