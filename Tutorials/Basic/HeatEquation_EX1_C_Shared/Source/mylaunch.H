#ifndef MY_LAUNCH_H_
#define MY_LAUNCH_H_

namespace amrex
{

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

template <typename L>
amrex::EnableIf_t<MaybeDeviceRunnable<L>::value>
ParallelForSMjki (Box const& box, L&& f) noexcept
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
            int i =  icell /   (len.y*len.z);
            int k = (icell - i*(len.y*len.z)) /   len.y;
            int j = (icell - i*(len.y*len.z)) - k*len.y;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            f(i,j,k);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

template <typename L>
amrex::EnableIf_t<MaybeDeviceRunnable<L>::value>
ParallelForSMkij (Box const& box, L&& f) noexcept
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
            int j =  icell /   (len.x*len.z);
            int i = (icell - j*(len.x*len.z)) /   len.z;
            int k = (icell - j*(len.x*len.z)) - i*len.z;
            i += lo.x;
            j += lo.y;
            k += lo.z;
            f(i,j,k);
        }
    });
    AMREX_GPU_ERROR_CHECK();
}

} // namespace amrex

#endif
