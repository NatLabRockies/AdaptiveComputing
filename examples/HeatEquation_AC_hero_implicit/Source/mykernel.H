#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>
#include <ac.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_phi (int i, int j, int k,
               amrex::Array4<amrex::Real> const& phi,
               GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM > 2)
    Real z = prob_lo[2] + (k+Real(0.5)) * dx[2];
    Real r2 = ((x-Real(0.25))*(x-Real(0.25))+(y-Real(0.25))*(y-Real(0.25))+(z-Real(0.25))*(z-Real(0.25)))/Real(0.01);
#else
    Real z = Real(0.);
    Real r2 = ((x-Real(0.25))*(x-Real(0.25))+(y-Real(0.25))*(y-Real(0.25)))/Real(0.01);
#endif
    phi(i,j,k) = Real(1.) + std::exp(-r2);
    phi(i,j,k) = Real(300.) + 100.*std::exp(-r2);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real conductivity(amrex::Real temperature
#ifndef AMREX_USE_GPU
			 , PyObject* ac_driver
#endif
  )
{
#ifndef AMREX_USE_GPU
  return conductivity_cpu(temperature, ac_driver);
#else
  // GPU / Fast Path
  if (d_kriging_model) {
    return std::max(1e-10, d_kriging_model->predict(temperature));
  }
  
  // Fallback if model not initialized
  return 16.0 + 0.01 * (temperature - 300.0);
#endif
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


#endif
