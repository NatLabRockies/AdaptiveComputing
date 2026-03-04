#include <AMReX_REAL.H>
#include <iostream>

AMREX_GPU_HOST_DEVICE
amrex::Real get_thermal_diffusivity(amrex::Real temperature) {
    
    // Simple linear model for temperature-dependent diffusivity
    amrex::Real alpha = 1e-6 + 1e-6 * temperature;

    std::cout << "T-Dependent Alpha calc called for T="<< temperature << std::endl;    
    // Ensure alpha is physically meaningful (non-negative)
    return std::max(1e-8, alpha);
}
