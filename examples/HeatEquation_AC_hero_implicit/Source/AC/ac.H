#ifndef _AC_H_
#define _AC_H_
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include <Python.h>
#include <numpy/arrayobject.h>

struct KrigingModel
{
  int n_samples;
  amrex::Real* X_norma;
  amrex::Real* alpha; // gamma
  amrex::Real* theta;
  amrex::Real beta;
  amrex::Real X_offset;
  amrex::Real X_scale;
  amrex::Real y_mean;
  amrex::Real y_std;
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  amrex::Real predict(amrex::Real x) const
    {
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

static AMREX_GPU_DEVICE_MANAGED KrigingModel* d_kriging_model = nullptr;

double get_double_from_entry(PyObject* py_array, int row, int col);
void set_double_at_entry(PyObject* py_array, int row, int col, double value);
amrex::Real* copy_numpy_to_device(PyObject* dict, const char* key, int& size);
amrex::Real get_scalar_from_dict(PyObject* dict, const char* key);
void update_gpu_model_from_python(PyObject* py_module, PyObject* ac_driver);

#endif
