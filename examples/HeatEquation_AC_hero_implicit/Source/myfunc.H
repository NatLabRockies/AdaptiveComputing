#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include "ac.H"

void main_main (PyObject *ac_driver);

void advance (amrex::MultiFab& phi_old,
              amrex::MultiFab& phi_new,
              amrex::Real dt,
              const amrex::Geometry& geom,
              const amrex::Vector<amrex::BCRec>& bc,
	      PyObject *ac_driver);

void init_phi (amrex::MultiFab& phi_new,
               amrex::Geometry const& geom);

amrex::Real conductivity_cpu(amrex::Real temperature
#ifndef AMREX_USE_GPU
			 , PyObject* ac_driver
#endif
  );
#endif
