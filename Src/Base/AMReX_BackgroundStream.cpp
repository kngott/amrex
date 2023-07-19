
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

void CUDART_CB amrex_elixir_delete (void* p)
{
    auto p_pa = reinterpret_cast<Vector<std::pair<void*,Arena*> >*>(p);
    for (auto const& pa : *p_pa) {
        pa.second->free(pa.first);
    }
    delete p_pa;
}

void
BackgroundStream::cpuSubmit (std::function<void()>&& f)
{
    BL_PROFILE("BGS::cpuSubmit");

    amrex::Print() << " cpuSubmit " << std::endl;

    if (previous == GPU) {

        {
            BL_PROFILE("BGS:cpuSubmit(GPU, prep)");
            std::lock_guard<std::mutex> guard(e_mtx);

            cudaEvent_t& place_event = events.emplace();
            AMREX_CUDA_SAFE_CALL(cudaEventCreateWithFlags(&place_event, cudaEventBlockingSync | cudaEventDisableTiming));
            AMREX_CUDA_SAFE_CALL(cudaEventRecord(place_event, gpu_stream));
        }

        cudaEvent_t& this_event = events.back();

        Submit( [=] ()
        {
            amrex::Print() << "Cpu Top" << std::endl;
            AMREX_GPU_ERROR_CHECK();

//            BL_PROFILE("BGS:cpuSubmit(GPU)");
            // Because a single thread is running through the functions in order,
            // the "front" first object in the queue should always be the next one needed.
            amrex::Print() << "Event Sync" << std::endl;
            AMREX_GPU_ERROR_CHECK();

            AMREX_CUDA_SAFE_CALL(cudaEventSynchronize(events.front()));

            amrex::Print() << "Event Sync Done" << std::endl;
            AMREX_GPU_ERROR_CHECK();

            {
                std::lock_guard<std::mutex> guard(e_mtx);
                AMREX_CUDA_SAFE_CALL(cudaEventDestroy(events.front()));
                events.pop();
            }

            amrex::Print() << "Event Cleanup" << std::endl;
            AMREX_GPU_ERROR_CHECK();

            f();

            amrex::Print() << "Event Finish" << std::endl;
            AMREX_GPU_ERROR_CHECK();

            amrex::Print() << "Event Done" << std::endl;
        });
    } else {
        Submit( std::move(f) );
    }

    previous = CPU;
}

void
BackgroundStream::gpuSubmit (std::function<void()>&& f)
{
    if (previous == CPU) {

        op_value++;

        int my_value = op_value;
        Submit( [=] ()
        {
           (*hptr) = my_value;
        });

       CU_CHECK(cuStreamWaitValue32_v2(gpu_stream, dptr, op_value, CU_STREAM_WAIT_VALUE_EQ));
   } 

    // Is a lambda over the ParallelFor function for now. Will needs lots of alternatives if don't want this.
    // Also include error check?
    f();

    previous = GPU;
}

void
BackgroundStream::gpuSubmit (std::function<void(amrex::gpuStream_t& s)>&& f)
{
    BL_PROFILE("BGS::gpuSubmit");
    amrex::Print() << " gpuSubmit " << std::endl;

    if (previous == CPU) {
        BL_PROFILE("BGS::gpuSubmit(CPU)");

        op_value++;

        int my_value = op_value;
        Submit( [=] ()
        {
           amrex::Print() << "gpuSubmit A " << std::endl;
           AMREX_GPU_ERROR_CHECK();

           (*hptr) = my_value;

           amrex::Print() << "gpuSubmit B " << std::endl;
           AMREX_GPU_ERROR_CHECK();
        });

       CU_CHECK(cuStreamWaitValue32_v2(gpu_stream, dptr, op_value, CU_STREAM_WAIT_VALUE_EQ));
   }

    // Is a lambda over the ParallelFor function for now. Will needs lots of alternatives if don't want this.
    // Also include error check?
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
    // Need a streamSync for a passed stream
    // amrex::Gpu::streamSynchronize(gpu_stream);

    AMREX_CUDA_SAFE_CALL(cudaStreamSynchronize(gpu_stream));

    if (previous == GPU) {
        previous = NONE;
    }
}

void
BackgroundStream::sync ()
{
    // Ordering?
    cpuSync();
    gpuSync();
}


} // amrex
