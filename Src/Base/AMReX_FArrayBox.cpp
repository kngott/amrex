
#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>

#include <AMReX_BLassert.H>
#include <AMReX.H>
#include <AMReX_Utility.H>
#include <AMReX_MemPool.H>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace amrex {

bool FArrayBox::initialized = false;

#if defined(AMREX_DEBUG) || defined(AMREX_TESTING)
bool FArrayBox::do_initval = true;
bool FArrayBox::init_snan  = true;
#else
bool FArrayBox::do_initval = false;
bool FArrayBox::init_snan  = false;
#endif
Real FArrayBox::initval;

static const char sys_name[] = "IEEE";

FArrayBox::FArrayBox () noexcept {}

FArrayBox::FArrayBox (Arena* ar) noexcept
    : BaseFab<Real>(ar)
{}

FArrayBox::FArrayBox (const Box& b, int ncomp, Arena* ar)
    : BaseFab<Real>(b,ncomp,ar)
{
    initVal();
}

FArrayBox::FArrayBox (const Box& b, int n, bool alloc, bool shared, Arena* ar)
    : BaseFab<Real>(b,n,alloc,shared,ar)
{
    if (alloc) initVal();
}

FArrayBox::FArrayBox (const FArrayBox& rhs, MakeType make_type, int scomp, int ncomp)
    : BaseFab<Real>(rhs,make_type,scomp,ncomp)
{
}

FArrayBox::FArrayBox (const Box& b, int ncomp, Real* p) noexcept
    : BaseFab<Real>(b,ncomp,p)
{
}

FArrayBox::FArrayBox (const Box& b, int ncomp, Real const* p) noexcept
    : BaseFab<Real>(b,ncomp,p)
{
}

void
FArrayBox::initVal () noexcept
{
    Real * p = dataPtr();
    Long s = size();
    if (p && s > 0) {
        RunOn runon;
#if defined(AMREX_USE_GPU)
        if (Gpu::inLaunchRegion() && arena()->isDeviceAccessible()) {
            runon = RunOn::Gpu;
        } else {
            runon = RunOn::Cpu;
        }
#else
        runon = RunOn::Cpu;
#endif

        if (init_snan) {
#if defined(AMREX_USE_GPU)
            if (runon == RunOn::Gpu)
            {
#if (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
                amrex::ParallelFor(s, [=] AMREX_GPU_DEVICE (Long i) noexcept
                {
                    p[i] = std::numeric_limits<Real>::signaling_NaN();
                });
#endif
                Gpu::streamSynchronize();
            }
            else
#endif
            {
                amrex_array_init_snan(p, s);
            }
        } else if (do_initval) {
            const Real x = initval;
            AMREX_HOST_DEVICE_PARALLEL_FOR_1D_FLAG (runon, s, i,
            {
                p[i] = x;
            });
            if (runon == RunOn::Gpu) Gpu::streamSynchronize();
        }
    }
}

void
FArrayBox::resize (const Box& b, int N, Arena* ar)
{
    BaseFab<Real>::resize(b,N,ar);
    initVal();
}

bool
FArrayBox::set_do_initval (bool tf)
{
    bool o_tf = do_initval;
    do_initval = tf;
    return o_tf;
}

bool
FArrayBox::get_do_initval ()
{
    return do_initval;
}

Real
FArrayBox::set_initval (Real iv)
{
    Real o_iv = initval;
    initval = iv;
    return o_iv;
}

Real
FArrayBox::get_initval ()
{
    return initval;
}

void
FArrayBox::Initialize ()
{
    if (initialized) return;
    initialized = true;

    BL_ASSERT(fabio == 0);

    ParmParse pp("fab");

    std::string fmt;
    //
    // This block sets ordering which doesn't affect output format.
    // It is only used when reading in an old FAB.
    //
    std::string ord;

    initval = std::numeric_limits<Real>::has_quiet_NaN
            ? std::numeric_limits<Real>::quiet_NaN()
            : std::numeric_limits<Real>::max();

    pp.query("initval",    initval);
    pp.query("do_initval", do_initval);
    pp.query("init_snan", init_snan);

    amrex::ExecOnFinalize(FArrayBox::Finalize);
}

void
FArrayBox::Finalize ()
{
    initialized = false;
}

//
// Copied from Utility.H.
//
#define BL_IGNORE_MAX 100000

}
