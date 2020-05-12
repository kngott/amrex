#ifndef MY_LAUNCH_H_
#define MY_LAUNCH_H_

template <typename L>
amrex::EnableIf_t<MaybeDeviceRunnable<L>::value>
ParallelForSMijk (Box const& box, L&& f) noexcept
{
    if (amrex::isEmpty(box)) return;
    int ncells = box.numPts();
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto ec = Gpu::ExecutionConfig(ncells);
    AMREX_LAUNCH_KERNEL(ec.numBlocks, ec.numThreads, 0, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE () noexcept {
        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            f(i,j,k);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}


#if 0
// These are here for reference
template <typename T, typename L, typename M=amrex::EnableIf_t<std::is_integral<T>::value> >
amrex::EnableIf_t<MaybeDeviceRunnable<L>::value>
ParallelForREFERENCE (T n, L&& f) noexcept
{
    if (amrex::isEmpty(n)) return;
    const auto ec = Gpu::ExecutionConfig(n);
    AMREX_LAUNCH_KERNEL(ec.numBlocks, ec.numThreads, 0, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE () noexcept {
        for (T i = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             i < n; i += stride) {
            f(i);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename L>
amrex::EnableIf_t<MaybeDeviceRunnable<L>::value>
ParallelForREFERENCE (Box const& box, L&& f) noexcept
{
    if (amrex::isEmpty(box)) return;
    int ncells = box.numPts();
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto ec = Gpu::ExecutionConfig(ncells);
    AMREX_LAUNCH_KERNEL(ec.numBlocks, ec.numThreads, 0, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE () noexcept {
        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            f(i,j,k);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename T, typename L, typename M=amrex::EnableIf_t<std::is_integral<T>::value> >
amrex::EnableIf_t<MaybeDeviceRunnable<L>::value>
ParallelForREFERENCE (Box const& box, T ncomp, L&& f) noexcept
{
    if (amrex::isEmpty(box)) return;
    int ncells = box.numPts();
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto ec = Gpu::ExecutionConfig(ncells);
    AMREX_LAUNCH_KERNEL(ec.numBlocks, ec.numThreads, 0, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE () noexcept {
        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride) {
            int k =  icell /   (len.x*len.y);
            int j = (icell - k*(len.x*len.y)) /   len.x;
            int i = (icell - k*(len.x*len.y)) - j*len.x;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            for (T n = 0; n < ncomp; ++n) {
                f(i,j,k,n);
            }
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

#endif

#endif
