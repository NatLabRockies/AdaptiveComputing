#include "myfunc.H"
#include "mykernel.H"

#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MY_UNIQUE_SYMBOL
#include "ac.H"

using namespace amrex;

void init_phi(amrex::MultiFab& phi_new, amrex::Geometry const& geom)
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    for (MFIter mfi(phi_new); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto const& phiNew = phi_new.array(mfi);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    init_phi(i, j, k, phiNew, dx, prob_lo);
                });
    }
}

#ifndef AMREX_USE_GPU
amrex::Real conductivity(amrex::Real temperature,
			 PyObject*   ac_driver)
{
#if 1
  // CPU Fallback to Python if model not ready or explicitly on CPU
  if (d_kriging_model == nullptr) {

    if (PyArray_API == NULL) {
      import_array();
    }
    npy_intp dims1pt[2] = {1,1};
    PyObject* x_query = PyArray_SimpleNew(2, dims1pt, NPY_DOUBLE);
    set_double_at_entry(x_query,0,0,temperature);
    
    // Call ac_driver.query, returns a numpy ndarray type object
    PyObject *y_query = PyObject_CallMethod(ac_driver, "query", "O,s,d", x_query, "absolute_variance", 1.e-9);
    if (y_query == NULL) {
      amrex::Abort("query failed");
      return 16.0; // Fallback
    }
    amrex::Real kappa = get_double_from_entry(y_query,0,0);
    Py_DECREF(y_query);
    Py_DECREF(x_query);
    return std::max(1e-10, kappa);
  }
#endif
  // Fallback if model not initialized
  return 16.0 + 0.01 * (temperature - 300.0);
}
#endif
