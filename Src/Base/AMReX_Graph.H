#ifndef AMREX_GRAPH_H_
#define AMREX_GRAPH_H_

#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>

#include <numeric>

// TODO: CHECK WHETHER SIZE OF ADDED OBJ/FAB/EDGE > 0.
// OR TODO: CHECK CONSISTENT SIZES.

// TODO: INTS OR LONGS?
// TODO: ADD AND UPDATE?

// TODO: FLAG FOR ARE THE WEIGHTS GLOBAL OR LOCAL?
// TODO: UNIQUE SCALING ON EACH RANK?
// TODO: SIZE OF COMM ITEMS THE SAME?
// TODO: CHECK WHERE IOPROC SHOULD BE DOING THE WORK ONLY.
//       E.G. GLOBAL WEIGHTS, NODE LIST DEF, ETC.

// TODO: LABELS AS INTVECTS (LOW & HIGH)
// TODO: FOR NL -- STORE BA & DM, PRINT FROM THAT?

namespace amrex
{
    class Graph
    {
        public:

        Graph() = default;
        ~Graph() = default;

        // Highest-level, user functions.
        // Add a fab (node) or comm operation (edge).

        template <typename FAB>
        void addFab(const FabArray<FAB>& fab,
                    const std::string& node_name,
                    const std::vector<double>& weights,
                    const std::string& wgts_name,
                    const Real scaling);

        template <typename FAB>
        void addFab(const FabArray<FAB>& fab,
                    const std::string& name);

        // ---------------------------

        // name, from, to, scaling + FB variables.
        // ***** SumBoundary? EnforcePeriodicity? Particle Comms.
        // OverWrite Sync.
        template <typename FAB>
        void addFB(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab,
                   bool cross = false)
        {
            addFB(name, nodes, scaling,
                  fab, 0, fab.nComp(), fab.nGrowVect(), Periodicity::NonPeriodic(), cross);
        }

        template <typename FAB>
        void addFB(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab,
                   const Periodicity& period, bool cross = false)
        {
            addFB(name, nodes, scaling,
                  fab, 0, fab.nComp(), fab.nGrowVect(), period, cross);
        }

        template <typename FAB>
        void addFB(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab,
                   const IntVect& nghost, Periodicity& period, bool cross = false)
        {
            addFB(name, nodes, scaling,
                  fab, 0, fab.nComp(), nghost, period, cross);
        }

        template <typename FAB>
        void addFB(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab,
                   const int scomp, const int ncomp, bool cross = false)
        {
            addFB(name, nodes, scaling,
                  fab, scomp, ncomp, fab.nGrowVect(), Periodicity::NonPeriodic(), cross);
        }

        template <typename FAB>
        void addFB(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab,
                   const int scomp, const int ncomp,
                   const Periodicity& period, bool cross = false)
        {
            addFB(name, nodes, scaling,
                  fab, scomp, ncomp, fab.nGrowVect(), period, cross);
        }

        template <typename FAB>
        void addFB(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab,
                   const int /* scomp */, const int ncomp,
                   const IntVect& nghost, const Periodicity& period, bool cross = false)
        {
            const FabArrayBase::CommMetaData& fb = fab.getFB(nghost, period, cross, false);
            addEdgeList(name, nodes, nodes, scaling, fb, ncomp);
        }

#if 0
    void ParallelCopy (const FabArray<FAB>& fa,
                       const Periodicity&   period = Periodicity::NonPeriodic(),
                       CpOp                 op = FabArrayBase::COPY)
    void ParallelCopy (const FabArray<FAB>& src,
                       int                  src_comp,
                       int                  dest_comp,
                       int                  num_comp,
                       const Periodicity&   period = Periodicity::NonPeriodic(),
                       CpOp                 op = FabArrayBase::COPY)
    void ParallelCopy (const FabArray<FAB>& src,
                       int                  src_comp,
                       int                  dest_comp,
                       int                  num_comp,
                       int                  src_nghost,
                       int                  dst_nghost,
                       const Periodicity&   period = Periodicity::NonPeriodic(),
                       CpOp                 op = FabArrayBase::COPY)
    void ParallelCopy (const FabArray<FAB>& src,
                       int                  src_comp,
                       int                  dest_comp,
                       int                  num_comp,
                       const IntVect&       src_nghost,
                       const IntVect&       dst_nghost,
                       const Periodicity&   period = Periodicity::NonPeriodic(),
                       CpOp                 op = FabArrayBase::COPY,
                       const FabArrayBase::CPC* a_cpc = nullptr);
#endif

/*

        void addPC( );
*/
        // ---------------------------

        // Lower-level functions.
        // General implementations.

        void addNodeList(const std::string& name,
                         const std::vector<int>& rank_array,
                         const std::size_t item_size,
                         const std::vector<std::string>& labels);

        void addEdgeList(const std::string& name,
                         const std::string& from_name,
                         const std::string& to_name,
                         const double scaling,
                         const FabArrayBase::CommMetaData& comm_data,
                         const int ncomp);

        // ---------------------------

        void addNodeWeight(const std::string& node_name,
                           const std::string& wgts_name,
                           const std::vector<double> wgts,
                           const double scaling);

        void addEdgeWeight(const std::string& edge_name,
                           const std::string& wgts_name,
                           const std::vector<double> wgts,
                           const double scaling);

        // ---------------------------

        // Print to file
        void print(const std::string& filename,
                   const int wgt_precision = 5,
                   const bool replace_file = false);

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
            int m_size = -1;

            std::size_t m_bytes_per_item = 0;
            std::vector<std::string> m_labels;
            std::vector<int> m_ranks;          // DM
//            std::vector<IntVect> m_lo;       // BA
//            std::vector<IntVect> m_hi;       // BA

            std::vector<Weights> m_wgts;
        };

        // Comm operation
        struct EdgeList {
            std::string m_name;
            int m_id = -1;
            int m_offset = -1;
            int m_size = -1;

            std::pair<std::string, std::string> m_mynodes;   // int,int & ID instead?
            std::vector<std::pair<int,int>> m_edgelist;
            std::vector<std::string> m_labels;

            std::vector<Weights> m_wgts;
        };

        private:

        template<class L>
        int get_index(const std::string& find, const L& list);

        template<class L>
        bool is_present(const std::string& find, const L& list);

        template<class L>
        bool not_present(const std::string& find, const L& list);

        int m_n_count = 0;
        int m_e_count = 0;

        std::vector<NodeList> m_nodes;
        std::vector<EdgeList> m_edges;
    };

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    template <typename FAB>
    void Graph::addFab(const FabArray<FAB>& fab,
                       const std::string& node_name,
                       const std::vector<double>& weights,
                       const std::string& wgts_name,
                       const Real scaling)
    {
        addFab(fab, node_name);
        addNodeWeight(node_name, wgts_name, weights, scaling);
    }


    template<class FAB>
    void Graph::addFab(const FabArray<FAB>& fab,
                       const std::string& name)
    {
        if (not_present(name, m_nodes))
        {
            // Build labels from box array.
            std::vector<std::string> labels(fab.size());
            const BoxArray& ba = fab.boxArray();

            for (int i=0; i<fab.size(); ++i)
            {
                std::ostringstream oss("\"", std::ios_base::ate);
                const Box& bx = ba[i];

                oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                labels[i] = oss.str();
            }

            const std::vector<int>& dm = fab.DistributionMap().ProcessorMap();
            const std::size_t type_size = sizeof(typename FAB::value_type);

            addNodeList(name, dm, type_size, labels);
        }
    }

    // --------------------------------

    void Graph::addNodeList(const std::string& name,
                            const std::vector<int>& rank_array,
                            const std::size_t item_size,
                            const std::vector<std::string>& labels)
    {
        if (not_present(name, m_nodes))
        {
            NodeList nl;
            nl.m_name = name;
            nl.m_id = m_nodes.size();
            nl.m_offset = m_n_count;
            nl.m_size = rank_array.size();
            m_n_count += rank_array.size();

            nl.m_ranks = rank_array;
            nl.m_labels = labels;
            nl.m_bytes_per_item = item_size;

            m_nodes.emplace_back(std::move(nl));
        }
    }

    void Graph::addEdgeList(const std::string& name,
                            const std::string& from_name,
                            const std::string& to_name,
                            const double scaling,
                            const FabArrayBase::CommMetaData& comm_data,
                            const int ncomp)

    {
        // label => snd box
        // weight => bytes sent

        if (not_present(name, m_edges))
        {
            EdgeList el;
            el.m_name = name;
            el.m_id = m_edges.size();
            el.m_offset = m_e_count;
            el.m_mynodes.first = from_name;
            el.m_mynodes.second = to_name;

            // Don't do recvs, so comms aren't duplicated.
            const int N_locs = comm_data.m_LocTags->size();
            const int N_snds = comm_data.m_SndTags->size();
            el.m_size = (N_locs + N_snds);
            m_e_count += el.m_size;
            el.m_edgelist.reserve(el.m_size);
            el.m_labels.reserve(el.m_size);

            std::vector<double> weights;
            weights.reserve(el.m_size);

            int from_id = get_index(from_name, m_nodes);
            int to_id = get_index(to_name, m_nodes);

            int offset_src = m_nodes[from_id].m_offset;
            int offset_dst = m_nodes[to_id].m_offset;
            int type_size = m_nodes[from_id].m_bytes_per_item;

            const auto& LocTags = comm_data.m_LocTags;
            const auto& SndTags = comm_data.m_SndTags;

            // Combination of multiple ranges would make this much nicer. :)
            // Or, subfunction/lambda function to do this once.
            for (const FabArrayBase::CopyComTag& cct : *LocTags) {
                std::ostringstream oss("\"", std::ios_base::ate);
                const Box& bx = cct.sbox;
                oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                el.m_edgelist.emplace_back(cct.srcIndex, cct.dstIndex);
                el.m_labels.emplace_back(oss.str());
                weights.emplace_back(bx.numPts()*type_size*ncomp);
            }

            for (const auto& kv: *SndTags) {
                for (const auto& cct : kv.second) {
                    std::ostringstream oss("\"", std::ios_base::ate);
                    const Box& bx = cct.sbox;
                    oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                    el.m_edgelist.emplace_back(cct.srcIndex, cct.dstIndex);
                    el.m_labels.emplace_back(oss.str());
                    weights.emplace_back(bx.numPts()*type_size*ncomp);
                }
            }

            m_edges.emplace_back(std::move(el));
            addEdgeWeight(name, "bytes", weights, scaling);

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
                // Check length of weights is correct
                AMREX_ASSERT(wgts.size() == m_nodes[nl].m_ranks.size() );

                Weights new_wgt;
                new_wgt.m_name = wgts_name;
                new_wgt.m_weights = wgts;
                new_wgt.m_scaling = scaling;

                m_nodes[nl].m_wgts.emplace_back(std::move(new_wgt));
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
                // Check length of weights is correct
                AMREX_ASSERT(wgts.size() == m_edges[el].m_edges.size() );

                Weights new_wgt;
                new_wgt.m_name = wgts_name;
                new_wgt.m_weights = wgts;
                new_wgt.m_scaling = scaling;

                m_edges[el].m_wgts.emplace_back(std::move(new_wgt));
            }
        }
    }

    // --------------------------------

    template<class L>
    int Graph::get_index(const std::string& find,
                         const L& list)
    {
        for(int i=0; i<list.size(); ++i) {
            if (list[i].m_name == find)
                { return i; }
        }
        return -1;
    }

    template<class L>
    bool Graph::is_present(const std::string& find,
                           const L& list)
    {
        return (get_index(find, list) != -1);
    }

    template<class L>
    bool Graph::not_present(const std::string& find,
                            const L& list)
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

    /*
    name = [...]
    name_labels = [...]
    name_weight = [...]
    name_weight_scaling =
    */
    void Graph::print(const std::string& filename,
                      const int wgt_precision,
                      const bool replace_file)    // std::rename ?
    {
/*
        if (ParallelDescriptor::IOProcessor())
        {
            return;
        }
*/
        // Node comms: (local weights).
        // or, all comms: (can later collapse and improve).

        amrex::PrintToFile file(filename);

        for (int nid=0; nid<m_nodes.size(); ++nid)
        {
            const NodeList& nl = m_nodes[nid];

            std::ostringstream oss_r(std::ios_base::ate);
            std::ostringstream oss_l(std::ios_base::ate);

            oss_r << nl.m_name << " = [";
            oss_l << nl.m_name << "_labels = [";

            for (int i=0; i<nl.m_ranks.size(); ++i) {
                oss_r << " " << std::to_string(nl.m_ranks[i]); // To ensure no round-off.
                oss_l << " " << nl.m_labels[i];

            }
            oss_r << "]\n";
            oss_l << "]\n";

            file << oss_r.str() << std::endl << oss_l.str() << std::endl;


            for (int w=0; w<nl.m_wgts.size(); ++w)
            {
                const Weights& wt = nl.m_wgts[w];
                std::string this_name = nl.m_name + "_" + wt.m_name;

                std::ostringstream oss_w(std::ios_base::ate);
                oss_w.precision(wgt_precision);

                oss_w << this_name << " = [";
                for (int i=0; i<nl.m_ranks.size(); ++i) {
                    oss_w << " " << wt.m_weights[i];
                }
                oss_w << "]\n" <<
                         this_name << "_scaling = " << wt.m_scaling;

                file << oss_w.str() << std::endl;
            }
        }

        file << std::endl << std::endl;

        // name = [  ]
        // name_src = from
        // name_dst = to
        // name_labels = [ ]
        // name_weight = [ ]
        // name_weight_scaling =

        // Edge comms: Pairs, labels, weights (unknown size)
/*
        for (int eid=0; eid<m_edges.size(); ++eid)
        {
            if (!ParallelDescriptor::IOProcessor())
            {

                Send bytes in pairs / labels / weights

                Sends
            }
            else
            {
                Recvs
            }
        }
*/

        for (int eid=0; eid<m_edges.size(); ++eid)
        {
            if (eid==0) {
                file << std::endl << std::endl;
            }

            const EdgeList& el = m_edges[eid];

            std::ostringstream oss_e(std::ios_base::ate);
            std::ostringstream oss_l(std::ios_base::ate);

            oss_e << el.m_name << " = [";
            oss_l << el.m_name << "_labels = [";

            for (int i=0; i<el.m_edgelist.size(); ++i) {
                const std::pair<int,int>& edge = el.m_edgelist[i];
                oss_e << " (" << std::to_string(edge.first) << ","
                              << std::to_string(edge.second) << ")";    // To ensure no round-off.
                oss_l << " " << el.m_labels[i];
            }
            oss_e << "]\n";
            oss_l << "]\n";

            file << oss_e.str() << std::endl
                 << el.m_name << "_src = " << el.m_mynodes.first << std::endl
                 << el.m_name << "_dst = " << el.m_mynodes.second << std::endl
                 << oss_l.str() << std::endl;

            for (int w=0; w<el.m_wgts.size(); ++w)
            {
                const Weights& wt = el.m_wgts[w];
                std::string this_name = el.m_name + "_" + wt.m_name;

                std::ostringstream oss_w(std::ios_base::ate);
                oss_w.precision(wgt_precision);

                oss_w << this_name << " = [";
                for (int i=0; i<el.m_edgelist.size(); ++i) {
                    oss_w << " " << wt.m_weights[i];
                }
                oss_w << "]\n" <<
                         this_name << "_scaling = " << wt.m_scaling;

                file << oss_w.str() << std::endl;
            }
        }
    }
}

#endif
