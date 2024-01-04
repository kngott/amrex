#ifndef AMREX_BACKGROUND_STREAM_H_
#define AMREX_BACKGROUND_STREAM_H_

#include <AMReX_BackgroundThread.H>
#include <AMReX_Config.H>
#include <AMReX_GpuControl.H>

#ifdef AMREX_USE_CUDA
#include <cuda.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <future>

#define CU_CHECK(stmt)                                                            \
    do {                                                                          \
        CUresult result = (stmt);                                                 \
        if (CUDA_SUCCESS != result) {                                             \
            const char *str;                                                      \
            CUresult str_result = cuGetErrorString(result, &str);                 \
            if (str_result != CUDA_SUCCESS)                                       \
                fprintf(stderr, "[%s:%d] cu failed with unknown error %d\n",      \
                        __FILE__, __LINE__, result);                              \
            else                                                                  \
                fprintf(stderr, "[%s:%d] cu failed with %s \n",                   \
                        __FILE__, __LINE__, str);                                 \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

namespace amrex {

class BackgroundStream
    :
    public BackgroundThread
{
public:
    enum func_type { NONE, CPU, GPU };

    BackgroundStream ();
    ~BackgroundStream ();
    BackgroundStream (BackgroundStream const&) = delete;
    BackgroundStream (BackgroundStream &&) = delete;
    BackgroundStream& operator= (BackgroundStream const&) = delete;
    BackgroundStream& operator= (BackgroundStream &&) = delete;

    void cpuSubmit (std::function<void()>&& f);
    void gpuSubmit (std::function<void(amrex::gpuStream_t& s)>&& f);

    void cpuSubmitwithDependency (std::function<void()>&& f,
                                      BackgroundStream& dep);
    void gpuSubmitwithDependency (std::function<void(amrex::gpuStream_t& s)>&& f,
                                      BackgroundStream& dep);

    cudaStream_t& get_stream() { return gpu_stream; }
    func_type get_previous() { return previous; }

    void setSleep (double &seconds) { poll_sleep = seconds; }

    void cpuSync ();
    void gpuSync ();
    void sync    ();

private:

    cudaStream_t gpu_stream;

    func_type previous = NONE;

    int op_value = 0;
    CUdeviceptr dptr;
    volatile int* hptr;

    double poll_sleep = 0;
};

}

#endif
