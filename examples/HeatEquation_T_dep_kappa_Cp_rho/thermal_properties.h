#include <AMReX_REAL.H>
#include <iostream>

AMREX_GPU_HOST_DEVICE
amrex::Real get_thermal_conductivity(amrex::Real temperature) {
    
    // Simple linear model for temperature-dependent diffusivity
    amrex::Real kappa = 16 + 0.01 * (temperature-400);

    //std::cout << "T-Dependent Alpha calc called for T="<< temperature << std::endl;    
    // Ensure kappa is physically meaningful (non-negative)
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
