#ifndef AMREX_GRAPH_H_
#define AMREX_GRAPH_H_

#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>

#include <numeric>

// TODO: CHECK WHETHER SIZE OF ADDED OBJ/FAB/EDGE > 0.
// OR TODO: CHECK CONSISTENT SIZES.
// TODO: INTS OR LONGS? (THROUGHOUT: VECTOR SIZES)
// TODO: ADD -> UPDATE CURRENT LISTS/WEIGHTS?
//       EDIT WEIGHT TO PROPERLY HANDLE UPDATES.
// TODO: UNIQUE SCALING ON EACH RANK?
// TODO: SIZE OF COMM ITEMS THE SAME?
// TODO: CHECK WHERE IOPROC SHOULD BE DOING THE WORK ONLY.
//       E.G. GLOBAL WEIGHTS, NODE LIST DEF, ETC.

// TODO: COMM OPTIMIZATION. COMPLETE MESS, BUT READABLE NOW.

// TODO: CHECKS IF THERE IS NO INFORMATION ON A RANK.

// TODO: ALLOW SETTING OF IO PROC.
//       (MULTIPLE SIMULTANEOUS GRAPHS ON DIFFERENT RANKS).
// TODO: GLOBAL SCALING INFORMATION? DON'T NEED TO COLLECT SCALING?
// TODO: ADD GRAPH VERBOSITY

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
                         const FabArrayBase& fab,
                         const std::size_t& item_size);

        void addEdgeList(const std::string& name,
                         const std::string& from_name,
                         const std::string& to_name,
                         const double scaling,
                         const FabArrayBase::CommMetaData& comm_data,
                         const int ncomp);

        // ---------------------------

        void addNodeWeight(const std::string& node_name,
                           const std::string& wgts_name,
                           const std::vector<double>& wgts,
                           const double scaling,
                           const bool local = false);

        void addEdgeWeight(const std::string& edge_name,
                           const std::string& wgts_name,
                           const std::vector<double>& wgts,
                           const double scaling,
                           const bool local = true);

        // ---------------------------

        // Move all local data graph's host_proc.
        Graph assemble();

        // Print to file
        void print_doit(const std::string& filename,
                        const int wgt_precision = 5,
                        const bool replace_file = false);

        // Print to file
        void print(const std::string& filename,
                   const int wgt_precision = 5,
                   const bool replace_file = false);

        // Clear graph
        void clear();

        // +++++++++++++++++++++++++++++++++++++++++++++++

#if 0
        // Another option for gathering scaling information.
        // Likely need a "local" graph and a "global" graph class.
        // Better solution to this?

        struct Scaling {
            double m_local;
            std::vector<double> m_global;
        }
#endif

        struct Weight {
            bool m_local = false;
            std::string m_name;
            std::vector<double> m_weights;
            std::vector<double> m_scaling = std::vector<double>(1, 0.0);
        };

        // FabArray -- Node = Box.
        struct NodeList {
            std::string m_name;
            int m_id = -1;
            int m_offset = -1;
            int m_size = -1;

            std::size_t m_bytes_per_item = 0;

            FabArrayBase m_fab;
            BoxArray m_ba;
            DistributionMapping m_dm;

            std::vector<Weight> m_wgts;
        };

        // Comm operation -- Edge = Box-to-Box data transfer.
        struct EdgeList {
            std::string m_name;
            int m_id = -1;
            int m_offset = -1;
            int m_size = -1;

            std::pair<std::string, std::string> m_mynodes;  // NodeList.m_id instead? Lookup name at print?
            std::vector<int> m_from;
            std::vector<int> m_to;
            std::vector<std::string> m_labels;              // Box for Mesh, # particles for Particles?

            std::vector<Weight> m_wgts;
        };

//        private:

        // -----------------
        // Lookup functions

        template<class L>
        int get_index(const std::string& find, const L& list);

        template<class L>
        bool is_present(const std::string& find, const L& list);

        template<class L>
        bool not_present(const std::string& find, const L& list);

        // ------------------
        // The Graph Data:

        bool m_assembled = true; 
        int m_rank = ParallelDescriptor::IOProcessorNumber();

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
            const std::size_t type_size = sizeof(typename FAB::value_type);
            addNodeList(name, fab, type_size);
        }
    }

    // --------------------------------

    template<class L>
    int Graph::get_index(const std::string& find,
                         const L& list)
    {
        for(unsigned int i=0; i<list.size(); ++i) {
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

}

#endif
