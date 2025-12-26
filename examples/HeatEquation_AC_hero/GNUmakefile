# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../amrex

DEBUG        = FALSE
USE_MPI      = FALSE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 3

#LIBRARY_LOCATIONS += /projects/bpms/rahimi/conda-env/AC_hero/lib
LIBRARY_LOCATIONS += /opt/homebrew/anaconda3/envs/AC_hero/lib

#INCLUDE_LOCATIONS += /projects/bpms/rahimi/conda-env/AC_hero/include/python3.11
INCLUDE_LOCATIONS += /opt/homebrew/anaconda3/envs/AC_hero/include/python3.11
INCLUDE_LOCATIONS += /opt/homebrew/anaconda3/envs/AC_hero/lib/python3.11/site-packages/numpy/_core/include
LIBRARIES += -lpython3.11

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules


# add :
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/bpms/rahimi/conda-env/AC_hero/lib
# export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/anaconda3/envs/AC_hero/lib
