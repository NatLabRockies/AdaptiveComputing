# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../amrex

DEBUG        = FALSE
USE_MPI      = FALSE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 3

LIBRARY_LOCATIONS += /projects/bpms/rahimi/conda-env/AC_hero/lib

INCLUDE_LOCATIONS += /projects/bpms/rahimi/conda-env/AC_hero/include/python3.11

LIBRARIES += -lpython3.11

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules


# add :
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/bpms/rahimi/conda-env/AC_hero/lib
