
#include <AMReX_BackgroundStream.H>
#include <AMReX_FBI.H>
#include <AMReX_PCI.H>

template <class FAB>
template <typename BUF, class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::FBEP_nowait_stream (int scomp, int ncomp, const IntVect& nghost,
                                   const Periodicity& period, bool cross,
                                   bool enforce_periodicity_only,
                                   bool override_sync)
{
    BL_PROFILE_SYNC_START_TIMED("SyncBeforeComms: FB");
    BL_PROFILE("FillBoundaryStream_nowait()");

    AMREX_ASSERT(!enforce_periodicity_only || !override_sync);

    // If no work is done, fbdq.front()->TheFB = nullptr
    std::unique_ptr<FBData<FAB>>& fbd = new_fbd();
    amrex::ignore_unused(fbd);

    bool work_to_do;
    if (enforce_periodicity_only) {
        work_to_do = period.isAnyPeriodic();
    } else if (override_sync) {
        work_to_do = (nghost.max() > 0) || !is_cell_centered();
    } else {
        work_to_do = nghost.max() > 0;
    }
    if (!work_to_do) { return; }

    const FB& TheFB = getFB(nghost, period, cross, enforce_periodicity_only, override_sync);

    if (ParallelContext::NProcsSub() == 1)
    {
        //
        // There can only be local work to do.
        //
        int N_locs = (*TheFB.m_LocTags).size();
        if (N_locs == 0) return;
#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if defined(__CUDACC__) && defined(AMREX_USE_CUDA)
            if (Gpu::inGraphRegion())
            {
                FB_local_copy_cuda_graph_1(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp, true);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }

        return;
    }

#ifdef BL_USE_MPI

    //
    // Do this before prematurely exiting if running in parallel.
    // Otherwise sequence numbers will not match across MPI processes.
    //
    int SeqNum = ParallelDescriptor::SeqNum();

    const int N_locs = TheFB.m_LocTags->size();
    const int N_rcvs = TheFB.m_RcvTags->size();
    const int N_snds = TheFB.m_SndTags->size();

    if (N_locs == 0 && N_rcvs == 0 && N_snds == 0) {
        // No work to do.
        return;
    }

    fbd->fb    = &TheFB;
    fbd->scomp = scomp;
    fbd->ncomp = ncomp;
    fbd->tag   = SeqNum;

    //
    // Post rcvs. Allocate one chunk of space to hold'm all.
    //

    if (N_rcvs > 0) {
        BL_PROFILE("PostRcvs()");

        PostRcvs_stream<BUF>(*TheFB.m_RcvTags, fbd->the_recv_data,
                             fbd->recv_data, fbd->recv_size, fbd->recv_from, fbd->recv_reqs,
                             ncomp, SeqNum);
        fbd->recv_stat.resize(N_rcvs);
    }

    //
    // Post send's
    //
    char*&                          the_send_data = fbd->the_send_data;
    Vector<char*> &                     send_data = fbd->send_data;
    Vector<std::size_t>                 send_size;
    Vector<int>                         send_rank;
    Vector<MPI_Request>&                send_reqs = fbd->send_reqs;
    Vector<const CopyComTagsContainer*> send_cctc;

    if (N_snds > 0)
    {
        BL_PROFILE("PreparePackPostSnds()");

        PrepareSendBuffers_stream<BUF>(*TheFB.m_SndTags, the_send_data, send_data, send_size, send_rank,
                                       send_reqs, send_cctc, ncomp);

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if defined(__CUDACC__) && defined(AMREX_USE_CUDA)
            if (Gpu::inGraphRegion()) {
                FB_pack_send_buffer_cuda_graph(TheFB, scomp, ncomp, send_data, send_size, send_cctc);
            }
            else
#endif
            {
                pack_send_buffer_gpu<BUF>(*this, scomp, ncomp, send_data, send_size, send_cctc, true);
            }
        }
        else
#endif
        {
            pack_send_buffer_cpu<BUF>(*this, scomp, ncomp, send_data, send_size, send_cctc);
        }

        AMREX_ASSERT(send_reqs.size() == N_snds);
        PostSnds_stream(send_data, send_size, send_rank, send_reqs, fbd->tag);
    }

    FillBoundary_test_stream(fbd->recv_reqs, fbd->recv_stat);

    //
    // Do the local work.  Hope for a bit of communication/computation overlap.
    //
    if (N_locs > 0)
    {
        BL_PROFILE("Locals()");


#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if defined(__CUDACC__) && defined(AMREX_USE_CUDA)
            if (Gpu::inGraphRegion()) {
                FB_local_copy_cuda_graph_n(TheFB, scomp, ncomp);
            }
            else
#endif
            {
                FB_local_copy_gpu(TheFB, scomp, ncomp, true);
            }
        }
        else
#endif
        {
            FB_local_copy_cpu(TheFB, scomp, ncomp);
        }

        FillBoundary_test_stream(fbd->recv_reqs, fbd->recv_stat);
    }

#endif /*BL_USE_MPI*/
}

template <class FAB>
template <typename BUF, class F, typename std::enable_if<IsBaseFab<F>::value,int>::type Z>
void
FabArray<FAB>::FillBoundary_finish_stream ()
{
    std::unique_ptr<FBData<FAB>>& fbd = lookup_fbd();
    const FB* TheFB = fbd->fb;

    if (!TheFB) {
        n_filled = IntVect::TheZeroVector();
        delete_fbd();
        return;
    }

#ifdef AMREX_USE_MPI

    BL_PROFILE("FillBoundaryStream_finish()");
    BL_PROFILE_SYNC_STOP();

    const auto N_rcvs = static_cast<int>(TheFB->m_RcvTags->size());
    if (N_rcvs > 0)
    {
        Vector<const CopyComTagsContainer*> recv_cctc(N_rcvs,nullptr);
        for (int k = 0; k < N_rcvs; k++)
        {
            if (fbd->recv_size[k] > 0)
            {
                auto const& cctc = TheFB->m_RcvTags->at(fbd->recv_from[k]);
                recv_cctc[k] = &cctc;
            }
        }

        int actual_n_rcvs = N_rcvs - std::count(fbd->recv_data.begin(), fbd->recv_data.end(), nullptr);
        if (actual_n_rcvs > 0) {

            /* CPUSUBMIT THIS */
            Gpu::bg_stream().cpuSubmit( [&] {

                amrex::Print() << "Waitall Start" << std::endl;
                AMREX_GPU_ERROR_CHECK();
                amrex::Print() << "Waitall A" << std::endl;

                ParallelDescriptor::Waitall(fbd->recv_reqs, fbd->recv_stat);
#ifdef AMREX_DEBUG
                if (!CheckRcvStats(fbd->recv_stat, fbd->recv_size, fbd->tag))
                {
                    amrex::Abort("FillBoundary_finish failed with wrong message size");
                }
#endif
                amrex::Print() << "Waitall A" << std::endl;
                AMREX_GPU_ERROR_CHECK();
                amrex::Print() << "Waitall End" << std::endl;
            });
        }

        bool is_thread_safe = TheFB->m_threadsafe_rcv;

#ifdef AMREX_USE_GPU
        if (Gpu::inLaunchRegion())
        {
#if defined(__CUDACC__) && defined(AMREX_USE_CUDA)
            if (Gpu::inGraphRegion())
            {
                FB_unpack_recv_buffer_cuda_graph(*TheFB, fbd->scomp, fbd->ncomp,
                                                 fbd->recv_data, fbd->recv_size,
                                                 recv_cctc, is_thread_safe);
            }
            else
#endif
            {
                unpack_recv_buffer_gpu<BUF>(*this, fbd->scomp, fbd->ncomp, fbd->recv_data, fbd->recv_size,
                                            recv_cctc, FabArrayBase::COPY, is_thread_safe, true);
            }
        }
        else
#endif
        {
            unpack_recv_buffer_cpu<BUF>(*this, fbd->scomp, fbd->ncomp, fbd->recv_data, fbd->recv_size,
                                        recv_cctc, FabArrayBase::COPY, is_thread_safe);
        }

#if 0
        if (fbd->the_recv_data)
        {
            /* CPUSUBMIT THIS */
            Gpu::bg_stream().cpuSubmit( [=] {
                amrex::The_Comms_Arena()->free(fbd->the_recv_data);
                fbd->the_recv_data = nullptr;
            });
        }
#endif
    }

    const auto N_snds = static_cast<int>(TheFB->m_SndTags->size());

    Gpu::bg_stream().cpuSubmit( [&, N_snds, N_rcvs] {

        if ((N_rcvs > 0) && (fbd->the_recv_data)) {
            amrex::The_Comms_Arena()->free(fbd->the_recv_data);
            fbd->the_recv_data = nullptr;
        }

        if (N_snds > 0) {
            Vector<MPI_Status> stats(fbd->send_reqs.size());
            amrex::Print() << " WaitS Start" << std::endl;
            AMREX_GPU_ERROR_CHECK();
            amrex::Print() << " WaitS A " << std::endl;

            ParallelDescriptor::Waitall(fbd->send_reqs, stats);

            amrex::Print() << " WaitS B" << std::endl;
            AMREX_GPU_ERROR_CHECK();
            amrex::Print() << " WaitS Done" << std::endl;

            amrex::The_Comms_Arena()->free(fbd->the_send_data);
            fbd->the_send_data = nullptr;

            AMREX_GPU_ERROR_CHECK();
            amrex::Print() << " WaitS Done" << std::endl;

        }

        AMREX_GPU_ERROR_CHECK();
        amrex::Print() << " Delete fbd Start" << std::endl;

        delete_fbd();

        AMREX_GPU_ERROR_CHECK();
        amrex::Print() << " Delete fbd End" << std::endl;
    });

#if 0

    const auto N_snds = static_cast<int>(TheFB->m_SndTags->size());
    if (N_snds > 0) {

        /* CPUSUBMIT THIS */
        Vector<MPI_Status> stats(fbd->send_reqs.size());
        ParallelDescriptor::Waitall(fbd->send_reqs, stats);
        amrex::The_Comms_Arena()->free(fbd->the_send_data);
        fbd->the_send_data = nullptr;
    }

    /* CPUSUBMIT THIS */
    delete_fbd();
#endif

#endif
}

#ifdef BL_USE_MPI
template <class FAB>
template <typename BUF>
AMREX_NODISCARD TheFaArenaPointer
FabArray<FAB>::PrepareSendBuffers_stream (const MapOfCopyComTagContainers&     SndTags,
                                          Vector<char*>&                       send_data,
                                          Vector<std::size_t>&                 send_size,
                                          Vector<int>&                         send_rank,
                                          Vector<MPI_Request>&                 send_reqs,
                                          Vector<const CopyComTagsContainer*>& send_cctc,
                                          int                                  ncomp) const
{
    char* pointer = nullptr;
    PrepareSendBuffers<BUF>(SndTags, pointer, send_data, send_size, send_rank, send_reqs, send_cctc, ncomp);
    return TheFaArenaPointer(pointer);
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::PrepareSendBuffers_stream (const MapOfCopyComTagContainers&     SndTags,
                                          char*&                               the_send_data,
                                          Vector<char*>&                       send_data,
                                          Vector<std::size_t>&                 send_size,
                                          Vector<int>&                         send_rank,
                                          Vector<MPI_Request>&                 send_reqs,
                                          Vector<const CopyComTagsContainer*>& send_cctc,
                                          int                                  ncomp) const
{
    send_data.clear();
    send_size.clear();
    send_rank.clear();
    send_reqs.clear();
    send_cctc.clear();
    const auto N_snds = SndTags.size();
    if (N_snds == 0) return;
    send_data.reserve(N_snds);
    send_size.reserve(N_snds);
    send_rank.reserve(N_snds);
    send_reqs.reserve(N_snds);
    send_cctc.reserve(N_snds);

    Vector<std::size_t> offset; offset.reserve(N_snds);
    std::size_t total_volume = 0;
    for (auto const& kv : SndTags)
    {
        auto const& cctc = kv.second;

        std::size_t nbytes = 0;
        for (auto const& cct : kv.second)
        {
            nbytes += cct.sbox.numPts() * ncomp * sizeof(BUF);
        }

        std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
        nbytes = amrex::aligned_size(acd, nbytes); // so that bytes are aligned

        // Also need to align the offset properly
        total_volume = amrex::aligned_size(std::max(alignof(BUF), acd),
                                           total_volume);

        offset.push_back(total_volume);
        total_volume += nbytes;

        send_data.push_back(nullptr);
        send_size.push_back(nbytes);
        send_rank.push_back(kv.first);
        send_reqs.push_back(MPI_REQUEST_NULL);
        send_cctc.push_back(&cctc);
    }

    if (total_volume > 0)
    {
        the_send_data = static_cast<char*>(amrex::The_Comms_Arena()->alloc(total_volume));

        for (int i = 0, N = static_cast<int>(send_size.size()); i < N; ++i) {
            send_data[i] = the_send_data + offset[i];
        }
    } else {
        the_send_data = nullptr;
    }

}

template <class FAB>
void
FabArray<FAB>::PostSnds_stream (Vector<char*> const&       send_data,
                                Vector<std::size_t> const& send_size,
                                Vector<int> const&         send_rank,
                                Vector<MPI_Request>&       send_reqs,
                                int&                       SeqNum)
{
    /* CPUSUBMIT THIS */
    // Copy of send_size and send_rank unnecessary. Move &/or transfer ownership?
    // (This is the last place those vectors are used.)
    Gpu::bg_stream().cpuSubmit( [&, send_size, send_rank] {

        const auto N_snds = static_cast<int>(send_reqs.size());
        MPI_Comm comm = ParallelContext::CommunicatorSub();

        for (int j = 0; j < N_snds; ++j)
        {
            if (send_size[j] > 0) {
                const int rank = ParallelContext::global_to_local_rank(send_rank[j]);
                send_reqs[j] = ParallelDescriptor::Asend
                    (send_data[j], send_size[j], rank, SeqNum, comm).req();
            }
        }
    });
}

template <class FAB>
template <typename BUF>
TheFaArenaPointer FabArray<FAB>::PostRcvs_stream (const MapOfCopyComTagContainers&       RcvTags,
                          Vector<char*>&                         recv_data,
                          Vector<std::size_t>&                   recv_size,
                          Vector<int>&                           recv_from,
                          Vector<MPI_Request>&                   recv_reqs,
                          int                                    ncomp,
                          int                                    SeqNum) const
{
    char* pointer = nullptr;
    PostRcvs(RcvTags, pointer, recv_data, recv_size, recv_from, recv_reqs, ncomp, SeqNum);
    return TheFaArenaPointer(pointer);
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::PostRcvs_stream (const MapOfCopyComTagContainers&  RcvTags,
                                char*&                            the_recv_data,
                                Vector<char*>&                    recv_data,
                                Vector<std::size_t>&              recv_size,
                                Vector<int>&                      recv_from,
                                Vector<MPI_Request>&              recv_reqs,
                                int                               ncomp,
                                int                               SeqNum) const
{
    recv_data.clear();
    recv_size.clear();
    recv_from.clear();
    recv_reqs.clear();

    Vector<std::size_t> offset;
    std::size_t TotalRcvsVolume = 0;
    for (const auto& kv : RcvTags) // loop over senders
    {
        std::size_t nbytes = 0;
        for (auto const& cct : kv.second)
        {
            nbytes += cct.dbox.numPts() * ncomp * sizeof(BUF);
        }

        std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
        nbytes = amrex::aligned_size(acd, nbytes);  // so that nbytes are aligned

        // Also need to align the offset properly
        TotalRcvsVolume = amrex::aligned_size(std::max(alignof(BUF),acd),
                                              TotalRcvsVolume);

        offset.push_back(TotalRcvsVolume);
        TotalRcvsVolume += nbytes;

        recv_data.push_back(nullptr);
        recv_size.push_back(nbytes);
        recv_from.push_back(kv.first);
        recv_reqs.push_back(MPI_REQUEST_NULL);
    }

    const auto nrecv = static_cast<int>(recv_from.size());

    MPI_Comm comm = ParallelContext::CommunicatorSub();

    if (TotalRcvsVolume == 0)
    {
        the_recv_data = nullptr;
    }
    else
    {
        the_recv_data = static_cast<char*>(amrex::The_Comms_Arena()->alloc(TotalRcvsVolume));

        for (int i = 0; i < nrecv; ++i)
        {
            recv_data[i] = the_recv_data + offset[i];
            if (recv_size[i] > 0)
            {
                const int rank = ParallelContext::global_to_local_rank(recv_from[i]);
                recv_reqs[i] = ParallelDescriptor::Arecv
                    (recv_data[i], recv_size[i], rank, SeqNum, comm).req();
            }
        }
    }
}
#endif

template <class FAB>
void
FabArray<FAB>::FillBoundary_test_stream (Vector<MPI_Request>& reqs, Vector<MPI_Status>& stat)
{
    amrex::ignore_unused(reqs, stat);

#if defined(AMREX_USE_MPI) && !defined(AMREX_DEBUG)
    // Don't know why, but GNU captures reqs as Vector<int> unless directly
    // capturing them like this. So, this avoids what I assume is a compiler bug.
    Gpu::bg_stream().cpuSubmit( [=, &reqs, &stat] {

        amrex::Print() << "FB_test begin" << std::endl;

        // We only test if no DEBUG because in DEBUG we check the status later.
        // If Test is done here, the status check will fail.
        int flag;
        ParallelDescriptor::Test(reqs, flag, stat);

        amrex::Print() << "FB_test done" << std::endl;
    });
#endif
}

