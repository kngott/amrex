#ifndef AMREX_DotGraph_H_
#define AMREX_DotGraph_H_

#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>

#include <numeric>

namespace amrex
{
    void dot_graph (const std::string& filename,
                    const DistributionMapping& dm,
                    const Vector<Real>& weights=Vector<Real>());

    void dot_graph_python (const std::string& filename,
                           const DistributionMapping& dm,
                           const Vector<Real>& weights=Vector<Real>());

    template <class FAB>
    void dot_graph_raw(const std::string& filename,
                       const FabArray<FAB>& fab,
                       const Vector<Real>& weights=Vector<Real>(),
                       const DistributionMapping& dm_prev = DistributionMapping());

    template <class FAB>
    void dot_graph_raw(const std::string& filename,
                       const FabArray<FAB>& fab,
                       const Vector<Real>& weights,
                       const FabArrayBase::FB& fb,
                       const DistributionMapping& dm_prev = DistributionMapping());

    // ===========================================================
    // ===========================================================
    // ===========================================================

    template <class FAB>
    void dot_graph_raw(const std::string& filename,
                       const FabArray<FAB>& fab,
                       const Vector<Real>& weights,
                       const DistributionMapping& dm_prev)
    {
        if (ParallelDescriptor::IOProcessor())
        {
            const DistributionMapping& dm = fab.DistributionMap();
            const BoxArray& ba = fab.boxArray();

            int nbxs = dm.size();

            std::ostringstream dm_str    ("dm = [",      std::ios_base::ate);
            std::ostringstream dmprev_str("dm_prev = [", std::ios_base::ate);
            std::ostringstream ba_str    ("ba = [",      std::ios_base::ate);
            std::ostringstream wgts_str  ("wgts = [",    std::ios_base::ate);

            dm_str.precision(5);
            dmprev_str.precision(5);
            ba_str.precision(5);
            wgts_str.precision(5);

            for (int i=0; i<nbxs; ++i)
            {
                dm_str     << dm[i];
                dmprev_str << dm_prev[i];
                wgts_str   << weights[i];

                const Box& bx = ba[i];
                ba_str << "\"" << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                if (i != nbxs-1)
                {
                    dm_str     << ", ";
                    dmprev_str << ", ";
                    wgts_str   << ", ";
                    ba_str     << ", ";
                }
            }

            dm_str     << "]";
            dmprev_str << "]";
            ba_str     << "]";
            wgts_str   << "]";

            amrex::PrintToFile file("rawgraph."+filename);
            file << ParallelDescriptor::NProcs() << " " << nbxs << "\n"
                 << dm_str.str() << "\n\n" << wgts_str.str() << "\n\n"
                 << dmprev_str.str() << "\n\n" << ba_str.str();
        }
    }


    template <class FAB>
    void dot_graph_raw(const std::string& filename,
                       const FabArray<FAB>& fab,
                       const Vector<Real>& weights,
                       const FabArrayBase::FB& fb,
                       const int& ncomp,
                       const DistributionMapping& dm_prev)
    {

        amrex::PrintToFile file("rawgraph."+filename);

        if (ParallelDescriptor::IOProcessor())
        {
            const DistributionMapping& dm = fab.DistributionMap();
            const BoxArray& ba = fab.boxArray();

            int nbxs = dm.size();

            std::ostringstream dm_str    ("dm = [",      std::ios_base::ate);
            std::ostringstream dmprev_str("dm_prev = [", std::ios_base::ate);
            std::ostringstream ba_str    ("ba = [",      std::ios_base::ate);
            std::ostringstream wgts_str  ("wgts = [",    std::ios_base::ate);

//            dm_str.precision(5);
//            dmprev_str.precision(5);
//            ba_str.precision(5);
            wgts_str.precision(5);

            for (int i=0; i<nbxs; ++i)
            {
                dm_str     << dm[i];
                dmprev_str << dm_prev[i];
                wgts_str   << weights[i];

                const Box& bx = ba[i];
                ba_str << "\"" << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                if (i != nbxs-1)
                {
                    dm_str     << ", ";
                    dmprev_str << ", ";
                    wgts_str   << ", ";
                    ba_str     << ", ";
                }
            }

            dm_str     << "]";
            dmprev_str << "]";
            ba_str     << "]";
            wgts_str   << "]";

            file << ParallelDescriptor::NProcs() << " " << nbxs << "\n\n"
                 << dm_str.str() << "\n\n" << wgts_str.str() << "\n\n"
                 << dmprev_str.str() << "\n\n" << ba_str.str() << "\n\n";
        }

        std::ostringstream comms_str("", std::ios_base::ate);
//        comms_str.precision(5);

        {    // Locals
            const std::size_t type_size = sizeof(typename FAB::value_type);

            auto const& LocTags = fb.m_LocTags;
            int N_locs = LocTags->size();

            for (int i=0; i<N_locs; ++i) {
                const FabArrayBase::CopyComTag& tag = (*LocTags)[i];

                std::size_t nbytes = type_size * tag.sbox.numPts() * ncomp;

                comms_str << "\"" << tag.srcIndex << " " << tag.dstIndex << " " << nbytes << " "
                          << tag.sbox.smallEnd() << " " << tag.sbox.bigEnd() << " "
                          << tag.dbox.smallEnd() << " " << tag.dbox.smallEnd() << "\"";

                if (i != N_locs-1)
                {
                    comms_str << ", ";
                }
            }

            // Sends
            auto const& SndTags = fb.m_SndTags;
            int N_snds = SndTags->size();

            if ((N_locs > 0) && (N_snds > 0))
                { comms_str << ", "; }

            for (auto kv = SndTags->begin(); kv != SndTags->end(); ++kv) {
                auto const& cctc = kv->second;

                for (int i=0; i<cctc.size(); ++i) {
                    auto const& tag = cctc[i];

                    std::size_t nbytes = type_size * tag.sbox.numPts() * ncomp;
                    std::size_t acd = ParallelDescriptor::alignof_comm_data(nbytes);
                    nbytes = amrex::aligned_size(acd, nbytes);

                    comms_str << "\"" << tag.srcIndex << " " << tag.dstIndex << " " << nbytes << " " 
                              << tag.sbox.smallEnd() << " " << tag.sbox.bigEnd() << " "
                              << tag.dbox.smallEnd() << " " << tag.dbox.bigEnd() << "\"";

                    if ((std::next(kv, 1) != SndTags->end()) || (i != cctc.size()-1))
                    {
                        comms_str << ", ";
                    }
                }
            }
        }

        std::string my_comms = comms_str.str();
        std::size_t comm_size = my_comms.size();
        std::vector<std::size_t> gsize = ParallelDescriptor::Gather(comm_size,
                                                  ParallelDescriptor::IOProcessorNumber());

        std::vector<std::size_t> offset(ParallelDescriptor::NProcs(), 0);

        std::size_t total = gsize[0];
        for (int i=1; i<gsize.size(); ++i) {
            offset[i] = offset[i-1] + gsize[i-1];

            total += gsize[i];
        }

        std::string full_comms;
        full_comms.capacity(total);

        ParallelDescriptor::Gatherv( );

        if (ParallelDescriptor::IOProcessor()) {
              file << comms_str.str() << "\n\n";
        }
    }
}

#endif
