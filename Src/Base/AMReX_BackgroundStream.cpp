
#include <AMReX_BLProfiler.H>
#include <AMReX_BackgroundStream.H>
#include <AMReX_GpuError.H>
#include <AMReX_Arena.H>
#include <AMReX_Print.H>

namespace amrex {

BackgroundStream::BackgroundStream ()
{
    AMREX_CUDA_SAFE_CALL(cudaStreamCreate(&gpu_stream));

    AMREX_CUDA_SAFE_CALL(cudaHostAlloc((void**) &hptr, sizeof(int), cudaHostAllocMapped));
    CU_CHECK(cuMemHostGetDevicePointer(&dptr, (void*) hptr, 0));
    *(hptr) = 0;
}

BackgroundStream::~BackgroundStream ()
{
    // Wait for everything to finish.
    // Just the GPU part. (BGThread destructor should cover CPU)
    AMREX_CUDA_SAFE_CALL(cudaStreamSynchronize(gpu_stream));

    AMREX_CUDA_SAFE_CALL(cudaStreamDestroy(gpu_stream));
    AMREX_CUDA_SAFE_CALL(cudaFreeHost((void*) hptr));
}

void
BackgroundStream::cpuSubmit (std::function<void()>&& f)
{
    BL_PROFILE("BGS::cpuSubmit");

    if (previous == GPU) {
        op_value++;

        CU_CHECK(cuStreamWriteValue32_v2(gpu_stream, dptr, op_value, CU_STREAM_WRITE_VALUE_DEFAULT));

        const int my_value = op_value;
        Submit( [=] ()
        {
            // Poll here for value to change. Better option?
            while(*hptr != my_value) {
                amrex::Sleep(poll_sleep);
            }

            f();
        });
    } else {
        Submit( std::move(f) );
    }

    previous = CPU;
}

void
BackgroundStream::cpuSubmitwithDependency (std::function<void()>&& f, BackgroundStream& dep)
{
    BL_PROFILE("BGS::cpuSubmitwithDependency");

    auto dep_prev = dep.get_previous();

    if (dep_prev == GPU) {
        op_value++;

        CU_CHECK(cuStreamWriteValue32_v2(dep.get_stream(), dptr, op_value, CU_STREAM_WRITE_VALUE_DEFAULT));

        const int my_value = op_value;
        Submit( [=] ()
        {
            // Poll here for value to change. Better option?
            while(*hptr != my_value) {
                amrex::Sleep(poll_sleep);
            }

            f();
        });

    } else if ((dep_prev == CPU) && (&dep != this)) {
        // This is CPU -> CPU, so can use other methods,
        //      i.e. triggering rather than polling.

        op_value++;

        const int my_value = op_value;

        dep.Submit( [=] ()
        {
           (*hptr) = my_value;
        });

        Submit( [=] ()
        {
            // Poll here for value to change. Better option?
            while(*hptr != my_value) {
                amrex::Sleep(poll_sleep);
            }

            f();
        });

    } else {
        Submit( std::move(f) );
    }

    previous = CPU;
}

void
BackgroundStream::gpuSubmit (std::function<void(amrex::gpuStream_t& s)>&& f)
{
    BL_PROFILE("BGS::gpuSubmit");

    if (previous == CPU) {
        BL_PROFILE("BGS::gpuSubmit(CPU)");

        op_value++;

        const int my_value = op_value;
        Submit( [=] ()
        {
           (*hptr) = my_value;
        });

        CU_CHECK(cuStreamWaitValue32_v2(gpu_stream, dptr, op_value, CU_STREAM_WAIT_VALUE_EQ));
    }

    // Is a lambda over the ParallelFor function for now. Will needs lots of alternatives to make it a direct GPU kernel launch.
    // This version moves any launch prep to here. Any problems?
    f( get_stream() );

    previous = GPU;
}

void
BackgroundStream::gpuSubmitwithDependency (std::function<void(amrex::gpuStream_t& s)>&& f, BackgroundStream& dep)
{
    BL_PROFILE("BGS::gpuSubmit");

    auto dep_prev = dep.get_previous();

    if (dep_prev == CPU) {
        BL_PROFILE("BGS::gpuSubmit(CPU)");

        op_value++;

        const int my_value = op_value;
        dep.Submit( [=] ()
        {
           (*hptr) = my_value;
        });

        CU_CHECK(cuStreamWaitValue32_v2(gpu_stream, dptr, op_value, CU_STREAM_WAIT_VALUE_EQ));

    } else if ((dep_prev == GPU) && (&dep != this)) {
        op_value++;

        CU_CHECK(cuStreamWriteValue32_v2(dep.get_stream(), dptr, op_value, CU_STREAM_WRITE_VALUE_DEFAULT));

        CU_CHECK(cuStreamWaitValue32_v2(gpu_stream, dptr, op_value, CU_STREAM_WAIT_VALUE_EQ));

    }
    // Is a lambda over the ParallelFor function for now. Will needs lots of alternatives to make it a direct GPU kernel launch.
    // This version moves any launch prep to here. Any problems?
    f( get_stream() );

    previous = GPU;
}

void
BackgroundStream::cpuSync ()
{
    Finish();

    if (previous == CPU) {
        previous = NONE;
    }
}

void
BackgroundStream::gpuSync ()
{
    // Need an AMReX streamSync for a passed stream? e.g:
    // amrex::Gpu::streamSynchronize(gpu_stream);

    AMREX_CUDA_SAFE_CALL(cudaStreamSynchronize(gpu_stream));

    if (previous == GPU) {
        previous = NONE;
    }
}

void
BackgroundStream::sync ()
{
    // For now, do the last thing last.
    // Gives time for other sync to complete while work is completed.

    // Need both? Or just the last one?

    if (previous == CPU) {
        gpuSync();
        cpuSync();
    } else {
        cpuSync();
        gpuSync();
    }
}


} // amrex
