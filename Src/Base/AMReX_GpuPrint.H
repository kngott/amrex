#ifndef AMREX_GPU_PRINT_H_
#define AMREX_GPU_PRINT_H_

#include <AMReX_Config.H>

#if !defined(__APPLE__)

#include <cstdio>

#ifdef AMREX_USE_DPCPP
#include <CL/sycl.hpp>
#endif

// AMREX_PRINTF
#if (AMREX_DEVICE_COMPILE)
#if defined(AMREX_USE_DPCPP)
#  define AMREX_PRINTF(format,...) { \
      static const __attribute__((opencl_constant)) char amrex_i_format[] = format; \
      sycl::ONEAPI::experimental::printf(amrex_i_format, __VA_ARGS__); }
#elif defined(AMREX_USE_HIP)
#  define AMREX_PRINTF(...) ::printf(__VA_ARGS__);
#elif defined(AMREX_USE_CUDA)
#  define AMREX_PRINTF(...) std::printf(__VA_ARGS__);
#endif
#else
#  define AMREX_PRINTF(...) std::printf(__VA_ARGS__);
#endif

// AMREX_DEVICE_PRINTF
#if (AMREX_DEVICE_COMPILE)
# define AMREX_DEVICE_PRINTF(...) AMREX_PRINTF(...)
#else
# define AMREX_DEVICE_PRINTF(...)
#endif

#endif  // !defined(__APPLE__)
#endif  // AMREX_GPU_PRINT_H_

