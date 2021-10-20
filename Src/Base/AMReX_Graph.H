#ifndef AMREX_GRAPH_H_
#define AMREX_GRAPH_H_

#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>

#include <numeric>

namespace amrex
{
    class Graph
    {
        public:

        Graph();
        ~Graph();

        // Add -- if name exists, return.
        // Update -- if name exists, overwrite. Otherwise, add.

        // Add an objective. Matches based on string names.
        void addObj();
        void addObj();

        // Add an objective, matching already existing names.
        // Include string/weight combination for new weights.
        void addObj();
        void addObj();

        // Piecewise data addition. 
        // Can be used to add data without an objective or partial obj info. 
        void addNodeList();
        void addEdgeList();
        void addWeight();

        // Print to file
        void print(std::string filename);

        // Clear graph
        void clear();

        // +++++++++++++++++++++++++++++++++++++++++++++++ 

        struct Weights {
            std::string m_name;
            std::vector<double> m_weights;
            double m_scaling;
        };

        struct NodeList {
            std::string m_name;
            int m_id = -1;
            int m_offset = -1; 

            std::vector<int> m_ranklist;
            std::vector<std::string> m_labels;

            std::vector<Weights> m_wgts;
        };

        struct EdgeList {
            std::string m_name;
            int m_id = -1;
            int m_offset = -1;

            std::vector<std::pair<int,int>> m_edgelist;
            std::vector<std::string> m_labels;

            std::vector<Weights> m_wgts;
        };

        private:

        int m_n_count = 0;
        int m_e_count = 0;

        std::vector<NodeList> m_nodes;
        std::vector<EdgeList> m_edges;
    };

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    template<class FAB>
    void Graph::addNodeList(const FabArray<FAB>& fab,
                            const std::string& name)
    {
        if (not_present(name, m_nodes))
        {
            std::vector<std::string> labels(fab.size());
            BoxArray& ba = fab.boxArray();

            for (int i=0; i<fab.size(); ++i)
            {
                std::ostringstream oss("\"", std::ios_base:ate);
                const Box& bx = ba[i];

                oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                labels[i] = oss.str();
            }


            // Call other version, with ba = labels.
            addNodeList(fab.DistributionMapping(), name, labels);
        }
    }

    void Graph::addNodeList(const DistributionMapping& dm,
                            const std::string& name,
                            const std::vector<std::string>& labels = std::vector())
    {
        if (not_present(name, m_nodes))
        {
            NodeList nl;
            nl.m_name = name;
            nl.m_ranklist = dm.ProcessorMap();
            nl.m_labels = labels;

            nl.m_id = m_nodes.size();
            nl.m_offset = m_n_count;
            m_n_count += dm.size();

            nodes[nl].my_wgts.emplace_back(std::move(nl));
        }
    }

    void Graph::addEdgeList(const FabArrayBase::CommMetaData comm_data,
                            const std::string& name,
                            const std::vector<std::string> labels = std::vector())
    {
        if (not_present(name, m_edges))
        {
            
        }
    }
    // -------------------------------- 

    void Graph::addNodeWeight(const std::string& node_name,
                              const std::string& wgts_name,
                              const std::vector<double> wgts,
                              const double scaling)
    {
        int nl = get_index(node_name, m_nodes);

        if (nl != -1)
        {
            if (not_present(wgts_name, m_nodes[nl].m_wgts))
            {
                Weights wgts;
                wgts.m_name = wgts_name;
                wgts.m_weights = wgts;
                wgts.m_scaling = scaling;

                m_nodes[nl].m_wgts.emplace_back(std::move(wgts));
            }
        }
    }

    void Graph::addEdgeWeight(const std::string& edge_name,
                              const std::string& wgts_name,
                              const std::vector<double> wgts,
                              const double scaling)
    {
        int el = get_index(edge_name, m_edges);

        if (el != -1)
        {
            if (not_present(wgts_name, m_edges[el].m_wgts))
            {
                Weights wgts;
                wgts.m_name = wgts_name;
                wgts.m_weights = wgts;
                wgts.m_scaling = scaling;

                m_edges[el].m_wgts.emplace_back(std::move(wgts));
            }
        }
    }

    // -------------------------------- 

    template<class FAB>
    void Graph::addObj(const FabArray<FAB>& fab,
                       const std::string& node_name,
                       const std::vector<Real>& weights,
                       const Real scaling) {
    }

    void Graph::addObj(const DistributionMapping& dm,
                       const std::string& node_name,
                       const std::vector<Real>& weights
                       const Real scaling) {
    }

    void Graph::addObj(const std::string& noe_name, 
                       const std::string& wgt_name) {
    }

    // -------------------------------- 

    template<class SList>
    int Graph::get_index(const std::string& find,
                         const SList& list)
    {
        for(int i=0; i<list.size(); ++i) {
            if (list[i].name == find)
                { return i; }
        }
        return -1;
    }

    template<class SList>
    bool Graph::is_present(const std::string& find,
                           const SList& list)
    {
        return (get_index(find, list) != -1);
    }

    template<class SList>
    bool Graph::not_present(const std::string& find,
                            const SList& list)
    {
        return (get_index(find, list) == -1);
    }

    void Graph::clear()
    {
        m_n_count = 0;
        m_e_count = 0;
        m_nodes.clear();
        m_edges.clear();
    }

    // ===========================================================
    // ===========================================================
    // ===========================================================

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
