#ifndef AMREX_GRAPH_H_
#define AMREX_GRAPH_H_

#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>
#include <AMReX_iMultiFab.H>

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

// TODO: CLEAN UP LOCAL/GLOBAL MARKING METHODOLOGY

// TODO: COMM OPTIMIZATION. COMPLETE MESS, BUT READABLE NOW.

// TODO: CHECKS IF THERE IS NO INFORMATION ON A RANK.

// TODO: MARK ASSEMBLED GRAPHS
// TODO: CHOOSE WHAT TO RETURN TO NON-IO RANKS (BLANK)?

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

        // Add a node-list, sometimes including a weight,
        //   through an AMReX object. (FabArray, FabArrayBase, or LayoutData).

        template <typename FAB>
        void addFab(const FabArray<FAB>& fab,
                    const std::string& name);

        template <typename FAB>
        void addFab(const FabArray<FAB>& fab,
                    const std::string& node_name,
                    const std::vector<double>& weights,
                    const std::string& wgts_name,
                    const double scaling);

        void addLayoutData(const amrex::LayoutData<double>& ld,
                           const std::string& node_name,
                           const std::size_t data_size,
                           const std::string& wgts_name,
                           const double scaling);

        void addFab(const FabArrayBase& fab,
                    const std::string& name,
                    const size_t data_size);

        // ---------------------------

        // Highest-level, user functions.
        // Add a fab (node) or comm operation (edge).

        template <typename FAB>
        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                             const FabArray<FAB>& fab, bool cross = false, std::size_t buffer_type = 0)
        {
            addFillBoundary(name, nodes, scaling,
                            fab, 0, fab.nComp(), fab.nGrowVect(), Periodicity::NonPeriodic(), cross, buffer_type);
        }

        template <typename FAB>
        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                             const FabArray<FAB>& fab, const Periodicity& period, bool cross = false,
                             std::size_t buffer_type = 0)
        {
            addFillBoundary(name, nodes, scaling,
                            fab, 0, fab.nComp(), fab.nGrowVect(), period, cross, buffer_type);
        }

        template <typename FAB>
        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab, const IntVect& nghost, Periodicity& period,
                   bool cross = false, std::size_t buffer_type = 0)
        {
            addFillBoundary(name, nodes, scaling,
                            fab, 0, fab.nComp(), nghost, period, cross, buffer_type);
        }

        template <typename FAB>
        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab, const int scomp, const int ncomp, bool cross = false,
                   std::size_t buffer_type = 0)
        {
            addFillBoundary(name, nodes, scaling,
                            fab, scomp, ncomp, fab.nGrowVect(), Periodicity::NonPeriodic(), cross, buffer_type);
        }

        template <typename FAB>
        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab, const int scomp, const int ncomp,
                   const Periodicity& period, bool cross = false, std::size_t buffer_type = 0)
        {
            addFillBoundary(name, nodes, scaling,
                            fab, scomp, ncomp, fab.nGrowVect(), period, cross, buffer_type);
        }

        template <typename FAB>
        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                   const FabArray<FAB>& fab, const int /* scomp */, const int ncomp,
                   const IntVect& nghost, const Periodicity& period, bool cross = false,
                   std::size_t buffer_type = 0)
        {
            if (not_present(nodes, m_nodes)) { addFab(fab, nodes); }

            const FabArrayBase::CommMetaData& fb = fab.getFB(nghost, period, cross, false);
            addEdgeList(name, nodes, nodes, scaling, fb, ncomp, buffer_type);
        }

        void addFillBoundary(const std::string& name, const std::string& nodes, const double scaling,
                   const size_t type_size, const size_t comm_type_size,
                   const FabArrayBase& fab, const int /* scomp */, const int ncomp,
                   const IntVect& nghost, const Periodicity& period, bool cross = false)
        {
            if (not_present(nodes, m_nodes)) { addFab(fab, nodes, type_size); }

            const FabArrayBase::CommMetaData& fb = fab.getFB(nghost, period, cross, false);
            addEdgeList(name, nodes, nodes, scaling, fb, ncomp, comm_type_size);
        }

        // ..................

        template <typename FAB>
        void addParallelCopy(const std::string& name, const std::string& nodes_dst,
                             const std::string& nodes_src, const double scaling,
                             const FabArray<FAB>& dst, const FabArray<FAB>& src,
                             const Periodicity period = Periodicity::NonPeriodic(),
                             FabArrayBase::CpOp op = FabArrayBase::COPY,
                             const std::size_t buffer_size = 0)
        {
            addParallelCopy(name, nodes_dst, nodes_src, scaling,
                            src, dst, 0, 0, dst.nComp(), 0, 0, period, op,
                            nullptr, false, buffer_size);
        }

        template <typename FAB>
        void addParallelCopy(const std::string& name, const std::string& nodes_dst,
                             const std::string& nodes_src, const double scaling,
                             const FabArray<FAB>& dst, const FabArray<FAB>& src,
                             const int src_comp, const int dst_comp, const int num_comp,
                             const Periodicity period = Periodicity::NonPeriodic(),
                             FabArrayBase::CpOp op = FabArrayBase::COPY,
                             const std::size_t buffer_size = 0)
        {
            addParallelCopy(name, nodes_dst, nodes_src, scaling,
                            src, dst, src_comp, dst_comp, num_comp, 0, 0, period, op,
                            nullptr, false, buffer_size);
        }

        template <typename FAB>
        void addParallelCopy(const std::string& name, const std::string& nodes_dst,
                             const std::string& nodes_src, const double scaling,
                             const FabArray<FAB>& dst, const FabArray<FAB>& src,
                             const int src_comp, const int dst_comp, const int num_comp,
                             const int src_nghost, const int dst_nghost,
                             const Periodicity period = Periodicity::NonPeriodic(),
                             FabArrayBase::CpOp op = FabArrayBase::COPY,
                             const std::size_t buffer_size = 0)
        {
            addParallelCopy(name, nodes_dst, nodes_src, scaling,
                            src, dst, src_comp, dst_comp, num_comp,
                            IntVect(src_nghost), IntVect(dst_nghost), period, op,
                            nullptr, false, buffer_size);
        }

        template <typename FAB>
        void addParallelCopyToGhost(const std::string& name, const std::string& nodes_dst,
                                    const std::string& nodes_src, const double scaling,
                                    const FabArray<FAB>& dst, const FabArray<FAB>& src,
                                    const int src_comp, const int dst_comp, const int num_comp,
                                    const IntVect& src_nghost, const IntVect& dst_nghost,
                                    const Periodicity period = Periodicity::NonPeriodic(),
                                    const std::size_t buffer_size = 0)
        {
            addParallelCopy(name, nodes_dst, nodes_src, scaling,
                            src, dst, src_comp, dst_comp, num_comp,
                            src_nghost, dst_nghost, period, FabArrayBase::COPY,
                            nullptr, true, buffer_size);
        }

        template <typename FAB>
        void addParallelCopy(const std::string& name, const std::string& nodes_dst,
                             const std::string& nodes_src, const double scaling,
                             const FabArray<FAB>& dst, const FabArray<FAB>& src,
                             const int /*src_comp*/, const int /*dst_comp*/, const int num_comp,
                             const IntVect& src_nghost, const IntVect& dst_nghost,
                             const Periodicity period = Periodicity::NonPeriodic(),
                             FabArrayBase::CpOp /*op*/ = FabArrayBase::COPY,
                             const FabArrayBase::CPC* a_cpc = nullptr,
                             bool to_ghost_cells_only = false,
                             const std::size_t buffer_size = 0)
        {
            if (not_present(nodes_dst, m_nodes)) { addFab(dst, nodes_dst); }
            if (not_present(nodes_src, m_nodes)) { addFab(src, nodes_src); }

            const FabArrayBase::CommMetaData& pc = (a_cpc) ? *a_cpc :
                         dst.getCPC(dst_nghost, src, src_nghost, period, to_ghost_cells_only);
            addEdgeList(name, nodes_src, nodes_dst, scaling, pc, num_comp, buffer_size);
        }

        template <typename FAB>
        void addParallelCopy(const std::string& name, const std::string& nodes_dst,
                             const std::string& nodes_src, const double scaling,
                             const std::size_t& dst_type_size, const std::size_t& src_type_size,
                             const std::size_t& buffer_size,
                             const FabArrayBase& dst, const FabArrayBase& src,
                             const int /*src_comp*/, const int /*dst_comp*/, const int num_comp,
                             const IntVect& src_nghost, const IntVect& dst_nghost,
                             const Periodicity period = Periodicity::NonPeriodic(),
                             FabArrayBase::CpOp /*op*/ = FabArrayBase::COPY,
                             const FabArrayBase::CPC* a_cpc = nullptr,
                             bool to_ghost_cells_only = false)
        {
            if (not_present(nodes_dst, m_nodes)) { addFab(dst, nodes_dst, dst_type_size); }
            if (not_present(nodes_src, m_nodes)) { addFab(src, nodes_src, src_type_size); }

            const FabArrayBase::CommMetaData& pc = (a_cpc) ? *a_cpc :
                         dst.getCPC(dst_nghost, src, src_nghost, period, to_ghost_cells_only);
            addEdgeList(name, nodes_src, nodes_dst, scaling, pc, num_comp, buffer_size);
        }



        // ..................

        template <typename FAB>
        void addSumBoundary(const std::string& name, const std::string& nodes, const double scaling,
                            const FabArray<FAB>& fab,
                            const Periodicity& period = Periodicity::NonPeriodic())
        {
            addSumBoundary(name, nodes, scaling,
                           fab, 0, fab.nComp(), IntVect(0), period);
        }

        template <typename FAB>
        void addSumBoundary(const std::string& name, const std::string& nodes, const double scaling,
                            const FabArray<FAB>& fab,
                            const int scomp, const int ncomp,
                            const Periodicity& period = Periodicity::NonPeriodic())
        {
            addSumBoundary(name, nodes, scaling,
                           fab, scomp, ncomp, IntVect(0), period);
        }

        template <typename FAB>
        void addSumBoundary(const std::string& name, const std::string& nodes, const double scaling,
                            const FabArray<FAB>& fab,
                            const int scomp, const int ncomp,
                            const IntVect& nghost,
                            const Periodicity& period = Periodicity::NonPeriodic())
        {
            addSumBoundary(name, nodes, scaling,
                           fab, scomp, ncomp, fab.nGrowVect(), nghost, period);
        }

        template <typename FAB>
        void addSumBoundary(const std::string& name, const std::string& nodes, const double scaling,
                            const FabArray<FAB>& fab,
                            const int /*scomp*/, const int ncomp,
                            const IntVect& src_nghost, const IntVect& dst_nghost,
                            const Periodicity& period = Periodicity::NonPeriodic())
        {
            // If SumBoundary wouldn't be completed, don't create the edge list.
            //      (interpretted as: the comm doesn't happen, so don't add a list).
            if ( fab.nGrowVect() == IntVect::TheZeroVector()
              && fab.boxArray().ixType().cellCentered() )  { return; }

            if (not_present(nodes, m_nodes)) { addFab(fab, nodes); }

            const FabArrayBase::CommMetaData& pc =
                      fab.getCPC(dst_nghost, fab, src_nghost, period, false);
            addEdgeList(name, nodes, nodes, scaling, pc, ncomp);
        }

        // ..................

        template <typename FAB>
        void addEnforcePeriodicity(const std::string& name, const std::string& nodes, const double scaling,
                                   const FabArray<FAB>& fab, const Periodicity& period)
        {
            addEnforcePeriodicity(name, nodes, scaling, fab, 0, fab.nComp(), fab.nGrowVect(), period);
        }

        template <typename FAB>
        void addEnforcePeriodicity (const std::string& name, const std::string& nodes, const double scaling,
                                    const FabArray<FAB>& fab,
                                    int scomp, int ncomp, const Periodicity& period)
        {
            addEnforcePeriodicity(name, nodes, scaling, fab, scomp, ncomp, fab.nGrowVect(), period);
        }

        template <typename FAB>
        void addEnforcePeriodicity (const std::string& name, const std::string& nodes, const double scaling,
                                    const FabArray<FAB>& fab,
                                    int /*scomp*/, int ncomp, const IntVect& nghost,
                                    const Periodicity& period)
        {
            // If EnforcePeriodicity wouldn't be completed, don't create the edge list.
            //      (interpretted as: the comm doesn't happen, so don't add a list).
            if ( period.isAnyPeriodic() )
            {
                if (not_present(nodes, m_nodes)) { addFab(fab, nodes); }

                const FabArrayBase::CommMetaData& fb = fab.getFB(nghost, period, false, true);
                addEdgeList(name, nodes, nodes, scaling, fb, ncomp);
            }
        }

        // ..................

        //! Sync up nodal data with owners overriding non-owners
        template <typename FAB>
        void addOverrideSync (const std::string& name, const std::string& nodes, const double scaling,
                              const FabArray<FAB>& fab,
                              const Periodicity& period = Periodicity::NonPeriodic())
        {
            addOverrideSync(name, nodes, scaling, fab, iMultiFab(), period);
        }

        template <typename FAB>
        void addOverrideSync (const std::string& name, const std::string& nodes, const double scaling,
                              const FabArray<FAB>& fab,
                              const iMultiFab& /*msk*/, const Periodicity& period = Periodicity::NonPeriodic())
        {
            // If OverrideSync wouldn't be completed, don't create the edge list.
            //      (interpretted as: the comm doesn't happen, so don't add a list).
            if (fab.ixType().cellCentered()) { return; }

            if (not_present(nodes, m_nodes)) { addFab(fab, nodes); }

            // Zero ghost cell temporary, used in OverrideSync.
            FabArrayBase temp(fab.boxArray(), fab.DistributionMap(), fab.nComp(),
                              0, MFInfo(), fab.Factory());

            const FabArrayBase::CommMetaData& pc =
                      temp.getCPC(IntVect(0), fab, fab.nGrowVect(), period, false);
            addEdgeList(name, nodes, nodes, scaling, pc, fab.nComp());
        }

        // ---------------------------

        // Lower-level functions; general implementations.

        void addNodeList (const std::string& name,
                          const FabArrayBase& fab,
                          const std::size_t& item_size);

        void addEdgeList (const std::string& name,
                          const std::string& from_name,
                          const std::string& to_name,
                          const double scaling,
                          const FabArrayBase::CommMetaData& comm_data,
                          const int ncomp,
                          const std::size_t comm_size_type = 0);

        void appendEdgeList (const std::string& name,
                             const std::string& from_name,
                             const std::string& to_name,
                             const double scaling,
                             const FabArrayBase::CommMetaData& comm_data,
                             const int ncomp,
                             const std::size_t comm_size_type = 0);

        // ---------------------------

        template<class T>
        void addNodeWeight(const std::string& node_name,
                           const std::string& wgts_name,
                           const std::vector<T>& wgts,
                           const double scaling,
                           const bool local = false);

        void addNodeWeight(const std::string& node_name,
                           const std::string& wgts_name,
                           const LayoutData<double>& wgts,
                           const double scaling,
                           const bool local = false);

        void addEdgeWeight(const std::string& edge_name,
                           const std::string& wgts_name,
                           const std::vector<double>& wgts,
                           const double scaling,
                           const bool local = true);

        void appendEdgeWeight(const std::string& edge_name,
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

        // Print to file
        void print_table_doit(const std::string& dirname,
                              const int wgt_precision = 5,
                              const bool replace_file = false);

        // Print to file
        void print_table(const std::string& dirname,
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

            std::size_t m_comm_item_size = 0;

            // NodeList.m_id instead? Lookup name at print?
            std::pair<std::string, std::string> m_mynodes;
            std::vector<int> m_from;
            std::vector<int> m_to;
            std::vector<std::string> m_labels;  // Box for Mesh, # particles for Particles?

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

        bool m_assembled = false;
        int m_rank = ParallelDescriptor::IOProcessorNumber();

        int m_n_count = 0;
        int m_e_count = 0;

        // List of weights names for table output.
        std::vector<std::string> m_nwgts;
        std::vector<std::string> m_ewgts;

        std::vector<NodeList> m_nodes;
        std::vector<EdgeList> m_edges;
    };

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    template<class FAB>
    void Graph::addFab(const FabArray<FAB>& fab,
                       const std::string& name)
    {
        addFab(fab, name, sizeof(typename FAB::value_type));
    }

    template <typename FAB>
    void Graph::addFab(const FabArray<FAB>& fab,
                       const std::string& node_name,
                       const std::vector<double>& weights,
                       const std::string& wgts_name,
                       const double scaling)
    {
        // Avoid warning if fab already present, just check weight.
        if (not_present(node_name, m_nodes)) {
            addFab(fab, node_name, sizeof(typename FAB::value_type));
        }

        addNodeWeight(node_name, wgts_name, weights, scaling);
    }

    // --------------------------------

    template<class T>
    void Graph::addNodeWeight(const std::string& node_name,
                              const std::string& wgts_name,
                              const std::vector<T>& wgts,
                              const double scaling,
                              const bool local)
    {
        int nl = get_index(node_name, m_nodes);
        if (nl == -1) {
            amrex::Print() << " **** WARNING: "
                           << " Graph.addNodeWeight() called for a nodelist not in the graph -- "
                           << node_name << ". Returning." << std::endl;
            return;
        }
        if (is_present(wgts_name, m_nodes[nl].m_wgts)) {
            amrex::Print() << " **** WARNING: "
                           << " Graph.addNodeWeight() called for a weight already named in the graph -- "
                           << wgts_name << ". Returning." << std::endl;
            return;
        }

        // Check length of weights is correct (total or local)
        AMREX_ASSERT(long(wgts.size()) == m_nodes[nl].m_fab.size()
                  || long(wgts.size()) == m_nodes[nl].m_fab.local_size());

        Weight new_wgt;
        new_wgt.m_name = wgts_name;
        new_wgt.m_scaling[0] = scaling;
        new_wgt.m_local = local;

        new_wgt.m_weights.resize(wgts.size());
        for (unsigned long i=0; i<wgts.size(); ++i) {
            (new_wgt.m_weights)[i] = double(wgts[i]);
        }

        m_nodes[nl].m_wgts.emplace_back(std::move(new_wgt));

        m_nwgts.push_back(wgts_name);
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
