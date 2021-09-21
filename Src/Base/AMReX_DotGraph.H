#ifndef AMREX_DotGraph_H_
#define AMREX_DotGraph_H_

#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>

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
                       const Vector<Real>& weights,
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

        std::ostringstream comms("comm = [", std::ios_base::ate);

        auto const& LocTags = fb.m_LocTags;
        int N_locs = LocTags.size();
        for (int i=0; i<N_locs; ++i) {
            const CopyComTag& tag = LocTags[i];

        }

        // Gather & Gatherv

        if (IOPROC)
          { } // write to file;
    }



}

#endif
