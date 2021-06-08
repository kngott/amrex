
#ifndef BL_FARRAYBOX_H
#define BL_FARRAYBOX_H
#include <AMReX_Config.H>

#include <AMReX_Box.H>
#include <AMReX_BaseFab.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_FabFactory.H>
#include <cmath>

namespace amrex {

/**
* \brief  A Fortran Array of REALs
*
*  Fortran Array Box's (generally called FAB's) are objects constructed
*  to emulate the FORTRAN array.  Useful operations can be performed
*  upon FAB's in C++, and they provide a convenient interface to
*  FORTRAN when it is necessary to retreat into that language.
*
*  FArrayBox is derived from BaseFab<Real>.
*  FArrayBox adds additional useful capabilities which make sense
*  for Real types, such as I/O and L**p norms.
*
*  FArrayBox's may be output in various formats (see FABio above).
*
*  The format and precision may be set in a file read by the ParmParse
*  class by the "fab.format" variable.  Allowed values are:
*    ASCII
*    8BIT
*    NATIVE
*    NATIVE_32
*    IEEE32
*
*  FABs written using operator<< are always written in ASCII.
*  FABS written using writOn use the FABio::Format specified with
*  setFormat or the FABio::Format specified in the ParmParse file
*  read by init. If the FABio::Format is not set explicitly by either
*  of these two methods, then it defaults to NATIVE.
*
*  The C pre-processor macro AMREX_SPACEDIM must be defined to use
*  this class.  The internal precision of FARRAYBOX objects is
*  set by defining either BL_USE_FLOAT or BL_USE_DOUBLE
*
*  This class does NOT provide a copy constructor or assignment operator,
*  but it has a move constructor.
*/

class FArrayBox
    :
    public BaseFab<Real>
{
public:
    //! Construct an invalid FAB with no memory.
    FArrayBox () noexcept;

    explicit FArrayBox (Arena* ar) noexcept;

    FArrayBox (const Box& b, int ncomp, Arena* ar);

    /**
    * \brief Construct an initial FAB with the data space allocated but
    * not inititialized. ncomp is the number of components
    * (variables) at each data point in the Box.
    */
    explicit FArrayBox (const Box& b,
                        int        ncomp=1,
                        bool       alloc=true,
                        bool       shared=false,
                        Arena*     ar = nullptr);

    FArrayBox (const FArrayBox& rhs, MakeType make_type, int scomp, int ncomp);

    FArrayBox (const Box& b, int ncomp, Real const* p) noexcept;

    FArrayBox (const Box& b, int ncomp, Real* p) noexcept;

    explicit FArrayBox (Array4<Real> const& a) noexcept : BaseFab<Real>(a) {}

    FArrayBox (Array4<Real> const& a, IndexType t) noexcept : BaseFab<Real>(a,t) {}

    explicit FArrayBox (Array4<Real const> const& a) noexcept : BaseFab<Real>(a) {}

    explicit FArrayBox (Array4<Real const> const& a, IndexType t) noexcept : BaseFab<Real>(a,t) {}

    //!  The destructor.
    virtual ~FArrayBox () noexcept override {}

    FArrayBox (FArrayBox&& rhs) noexcept = default;

    FArrayBox (const FArrayBox&) = delete;
    FArrayBox& operator= (const FArrayBox&) = delete;
    FArrayBox& operator= (FArrayBox&&) = delete;

    //! Set the fab to the value r.
#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    FArrayBox& operator= (Real r) noexcept;

    /**
    * \brief Are there any NaNs in the FAB?
    * This may return false, even if the FAB contains NaNs, if the machine
    * doesn't support the appropriate NaN testing functions.
    */
#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_nan () const noexcept;

#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_nan (const Box& bx, int scomp, int ncomp) const noexcept;
    /**
    * \brief These versions return the cell index (though not the component) of
    * the first location of a NaN if there is one.
    */
#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_nan (IntVect& where) const noexcept;

#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_nan (const Box& bx, int scomp, int ncomp, IntVect& where) const noexcept;
    /**
    * \brief Are there any Infs in the FAB?
    * This may return false, even if the FAB contains Infs, if the machine
    * doesn't support the appropriate Inf testing functions.
    */
#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_inf () const noexcept;

#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_inf (const Box& bx, int scomp, int ncomp) const noexcept;
    /**
    * \brief These versions return the cell index (though not the component) of
    * the first location of an Inf if there is one.
    */
#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_inf (IntVect& where) const noexcept;

#if defined(AMREX_USE_GPU)
    template <RunOn run_on>
#else
    template <RunOn run_on=RunOn::Host>
#endif
    bool contains_inf (const Box& bx, int scomp, int ncomp, IntVect& where) const noexcept;

    //! For debugging purposes we hide BaseFab version and do some extra work.
    void resize (const Box& b, int N = 1, Arena* ar = nullptr);

    FabType getType () const noexcept { return m_type; }

    void initVal () noexcept; // public for cuda

    static bool set_do_initval (bool tf);
    static bool get_do_initval ();
    static Real set_initval    (Real iv);
    static Real get_initval    ();
    //! Initialize from ParmParse with "fab" prefix.
    static void Initialize ();
    static void Finalize ();
    static bool initialized;

protected:

    FabType m_type = FabType::regular;

    //! initial value
    static bool do_initval;
    static Real initval;
    static bool init_snan;
};

using FArrayBoxFactory = DefaultFabFactory<FArrayBox>;

template <RunOn run_on>
FArrayBox&
FArrayBox::operator= (Real v) noexcept
{
    BaseFab<Real>::operator=<run_on>(v);
    return *this;
}

template <RunOn run_on>
bool
FArrayBox::contains_nan () const noexcept
{
    const Real* dp = dptr;
    const Long n = numPts()*nvar;
#ifdef AMREX_USE_GPU
    if (run_on == RunOn::Device && Gpu::inLaunchRegion()) {
        ReduceOps<ReduceOpLogicalOr> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = ReduceData<int>::Type;
        reduce_op.eval(n, reduce_data,
        [=] AMREX_GPU_DEVICE (Long i) -> ReduceTuple
        {
            return { static_cast<int>(amrex::isnan(dp[i])) };
        });
        ReduceTuple hv = reduce_data.value(reduce_op);
        return amrex::get<0>(hv);
    } else
#endif
    {
        for (Long i = 0; i < n; i++) {
            if (amrex::isnan(*dp++)) {
                return true;
            }
        }
        return false;
    }
}

template <RunOn run_on>
bool
FArrayBox::contains_nan (const Box& bx, int scomp, int ncomp) const noexcept
{
    BL_ASSERT(scomp >= 0);
    BL_ASSERT(ncomp >= 1);
    BL_ASSERT(scomp <  nComp());
    BL_ASSERT(ncomp <= nComp());
    BL_ASSERT(domain.contains(bx));

    const auto& a = this->array();

#ifdef AMREX_USE_GPU
    if (run_on == RunOn::Device && Gpu::inLaunchRegion()) {
        ReduceOps<ReduceOpLogicalOr> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = ReduceData<int>::Type;
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            bool t = false;
            for (int n = scomp; n < scomp+ncomp; ++n) {
                t = t || amrex::isnan(a(i,j,k,n));
            }
            return { static_cast<int>(t) };
        });
        ReduceTuple hv = reduce_data.value(reduce_op);
        return amrex::get<0>(hv);
    } else
#endif
    {
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        for (int n = scomp; n < scomp+ncomp; ++n) {
            for         (int k = lo.z; k <= hi.z; ++k) {
                for     (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        if (amrex::isnan(a(i,j,k,n))) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}

template <RunOn run_on>
bool
FArrayBox::contains_nan (IntVect& where) const noexcept
{
    return contains_nan<run_on>(domain, 0, nComp(), where);
}

template <RunOn run_on>
bool
FArrayBox::contains_nan (const Box& bx, int scomp, int ncomp, IntVect& where) const noexcept
{
    BL_ASSERT(scomp >= 0);
    BL_ASSERT(ncomp >= 1);
    BL_ASSERT(scomp <  nComp());
    BL_ASSERT(ncomp <= nComp());
    BL_ASSERT(domain.contains(bx));

    const auto& a = this->array();
#ifdef AMREX_USE_GPU
    if (run_on == RunOn::Device && Gpu::inLaunchRegion()) {
        Array<int,1+AMREX_SPACEDIM> ha{0,AMREX_D_DECL(std::numeric_limits<int>::lowest(),
                                                      std::numeric_limits<int>::lowest(),
                                                      std::numeric_limits<int>::lowest())};
        Gpu::DeviceVector<int> dv(1+AMREX_SPACEDIM);
        int* p = dv.data();
        Gpu::htod_memcpy_async(p, ha.data(), sizeof(int)*ha.size());
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int* flag = p;
            bool t = false;
            for (int n = scomp; n < scomp+ncomp; ++n) {
                t = t || amrex::isnan(a(i,j,k,n));
            }
            if (t && (*flag == 0)) {
                if (Gpu::Atomic::Exch(flag,1) == 0) {
                    AMREX_D_TERM(p[1] = i;,
                                 p[2] = j;,
                                 p[3] = k;);
                }
            }
        });
        Gpu::dtoh_memcpy_async(ha.data(), p, sizeof(int)*ha.size());
        Gpu::streamSynchronize();
        where = IntVect(AMREX_D_DECL(ha[1],ha[2],ha[3]));
        return ha[0];
    } else
#endif
    {
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        for (int n = scomp; n < scomp+ncomp; ++n) {
            for         (int k = lo.z; k <= hi.z; ++k) {
                for     (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        if (amrex::isnan(a(i,j,k,n))) {
                            where = IntVect(AMREX_D_DECL(i,j,k));
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}

template <RunOn run_on>
bool
FArrayBox::contains_inf () const noexcept
{
    const Real* dp = dptr;
    const Long n = numPts()*nvar;
#ifdef AMREX_USE_GPU
    if (run_on == RunOn::Device && Gpu::inLaunchRegion()) {
        ReduceOps<ReduceOpLogicalOr> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = ReduceData<int>::Type;
        reduce_op.eval(n, reduce_data,
        [=] AMREX_GPU_DEVICE (Long i) -> ReduceTuple
        {
            return { static_cast<int>(amrex::isinf(dp[i])) };
        });
        ReduceTuple hv = reduce_data.value(reduce_op);
        return amrex::get<0>(hv);
    } else
#endif
    {
        for (Long i = 0; i < n; i++) {
            if (amrex::isinf(*dp++)) {
                return true;
            }
        }
        return false;
    }
}

template <RunOn run_on>
bool
FArrayBox::contains_inf (const Box& bx, int scomp, int ncomp) const noexcept
{
    BL_ASSERT(scomp >= 0);
    BL_ASSERT(ncomp >= 1);
    BL_ASSERT(scomp <  nComp());
    BL_ASSERT(ncomp <= nComp());
    BL_ASSERT(domain.contains(bx));

    const auto& a = this->array();

#ifdef AMREX_USE_GPU
    if (run_on == RunOn::Device && Gpu::inLaunchRegion()) {
        ReduceOps<ReduceOpLogicalOr> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = ReduceData<int>::Type;
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) ->ReduceTuple
        {
            bool t = false;
            for (int n = scomp; n < scomp+ncomp; ++n) {
                t = t || amrex::isinf(a(i,j,k,n));
            }
            return { static_cast<int>(t) };
        });
        ReduceTuple hv = reduce_data.value(reduce_op);
        return amrex::get<0>(hv);
    } else
#endif
    {
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        for (int n = scomp; n < scomp+ncomp; ++n) {
            for         (int k = lo.z; k <= hi.z; ++k) {
                for     (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        if (amrex::isinf(a(i,j,k,n))) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}

template <RunOn run_on>
bool
FArrayBox::contains_inf (IntVect& where) const noexcept
{
    return contains_inf<run_on>(domain,0,nComp(),where);
}

template <RunOn run_on>
bool
FArrayBox::contains_inf (const Box& bx, int scomp, int ncomp, IntVect& where) const noexcept
{
    BL_ASSERT(scomp >= 0);
    BL_ASSERT(ncomp >= 1);
    BL_ASSERT(scomp <  nComp());
    BL_ASSERT(ncomp <= nComp());
    BL_ASSERT(domain.contains(bx));

    const auto& a = this->array();
#ifdef AMREX_USE_GPU
    if (run_on == RunOn::Device && Gpu::inLaunchRegion()) {
        Array<int,1+AMREX_SPACEDIM> ha{0,AMREX_D_DECL(std::numeric_limits<int>::lowest(),
                                                      std::numeric_limits<int>::lowest(),
                                                      std::numeric_limits<int>::lowest())};
        Gpu::DeviceVector<int> dv(1+AMREX_SPACEDIM);
        int* p = dv.data();
        Gpu::htod_memcpy_async(p, ha.data(), sizeof(int)*ha.size());
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int* flag = p;
            bool t = false;
            for (int n = scomp; n < scomp+ncomp; ++n) {
                t = t || amrex::isinf(a(i,j,k,n));
            }
            if (t && (*flag == 0)) {
                if (Gpu::Atomic::Exch(flag,1) == 0) {
                    AMREX_D_TERM(p[1] = i;,
                                 p[2] = j;,
                                 p[3] = k;);
                }
            }
        });
        Gpu::dtoh_memcpy_async(ha.data(), p, sizeof(int)*ha.size());
        Gpu::streamSynchronize();
        where = IntVect(AMREX_D_DECL(ha[1],ha[2],ha[3]));
        return ha[0];
    } else
#endif
    {
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        for (int n = scomp; n < scomp+ncomp; ++n) {
            for         (int k = lo.z; k <= hi.z; ++k) {
                for     (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        if (amrex::isinf(a(i,j,k,n))) {
                            where = IntVect(AMREX_D_DECL(i,j,k));
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}

}

#endif /*BL_FARRAYBOX_H*/
