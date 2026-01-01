# AMREX_HOME defines the directory in which we will find all the AMReX code.
# AMREX_HOME ?= ../../../../../amrex
AMREX_HOME ?= ../../../../../../../CCSE/amrex

DEBUG     = FALSE
USE_MPI   = FALSE
USE_OMP   = FALSE
COMP      = gnu
COMP      = llvm
DIM       = 2

LIBRARY_LOCATIONS += $(CONDA_PREFIX)/lib
INCLUDE_LOCATIONS += $(CONDA_PREFIX)/include/python3.11
INCLUDE_LOCATIONS += $(CONDA_PREFIX)/lib/python3.11/site-packages/numpy/_core/include
LIBRARIES += -lpython3.11

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
include ../Source/AC/Make.package
VPATH_LOCATIONS  += ../Source ../Source/AC
INCLUDE_LOCATIONS += ../Source ../Source/AC

Pdirs 	:= Base Boundary LinearSolvers/MLMG
Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

# add :
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
# export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${CONDA_PREFIX}/lib
