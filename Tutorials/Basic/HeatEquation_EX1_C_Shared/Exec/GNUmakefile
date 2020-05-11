# AMREX_HOME defines the directory in which we will find all the AMReX code.
# If you set AMREX_HOME as an environment variable, this line will be ignored
AMREX_HOME ?= /ccs/home/kngott/amrex

DEBUG        = FALSE
USE_MPI      = TRUE 
MPI_MULTIPLE = FALSE  # amrex.async_out=1 with more than 64 processes requires MPI_MULTIPLE
USE_OMP      = FALSE
USE_CUDA     = TRUE
TINY_PROFILE = TRUE
COMP         = gnu
DIM          = 3
BL_NO_FORT   = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
