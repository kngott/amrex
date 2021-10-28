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
        template <typename FAB>
        void addObj(const FabArray<FAB>& fab,
                    const std::string& node_name,
                    const std::vector<double>& weights,
                    const std::string& wgts_name,
                    const Real scaling);

        void addObj(const DistributionMapping& dm,
                    const std::string& node_name,
                    const std::vector<std::string> labels,
                    const std::vector<double>& weights,
                    const std::string& wgts_name,
                    const Real scaling);

        // Piecewise data addition. 
        // Can be used to add data without an objective or partial obj info.
        template <typename FAB>
        void addNodeList(const FabArray<FAB>& fab,
                         const std::string& name);

        void addNodeList(const DistributionMapping& dm,
                         const std::string& name,
                         const std::vector<std::string>& labels = std::vector<std::string>());

//        void addEdgeList();
        void addNodeWeight(const std::string& node_name,
                           const std::string& wgts_name,
                           const std::vector<double> wgts,
                           const double scaling);

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

            std::vector<int> m_ranks;
            std::vector<std::string> m_labels;

            std::vector<Weights> m_wgts;
        };

        // Comm operation
        struct EdgeList {
            std::string m_name;
            int m_id = -1;
            int m_offset = -1;

            std::pair<std::string, std::string> m_mynodes;
            std::vector<std::pair<int,int>> m_edges;
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

    template<class FAB>
    void Graph::addNodeList(const FabArray<FAB>& fab,
                            const std::string& name)
    {
        if (not_present(name, m_nodes))
        {
            // Build labels from box array and move on.
            std::vector<std::string> labels(fab.size());
            BoxArray& ba = fab.boxArray();

            for (int i=0; i<fab.size(); ++i)
            {
                std::ostringstream oss("\"", std::ios_base::ate);
                const Box& bx = ba[i];

                oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                labels[i] = oss.str();
            }

            addNodeList(fab.DistributionMapping(), name, labels);
        }
    }

    void Graph::addNodeList(const DistributionMapping& dm,
                            const std::string& name,
                            const std::vector<std::string>& labels)
    {
        if (not_present(name, m_nodes))
        {
            NodeList nl;
            nl.m_name = name;
            nl.m_ranks = dm.ProcessorMap();
            nl.m_labels = labels;

            nl.m_id = m_nodes.size();
            nl.m_offset = m_n_count;
            m_n_count += dm.size();

            m_nodes.emplace_back(std::move(nl));
        }
    }
/*
    void Graph::addEdgeList(const std::string& name,
                            const FabArrayBase::CommMetaData comm_data,
                            const std::string& from_name,
                            const std::string& to_name,
                            const std::string& name,
                            const std::vector<std::string> labels = std::vector())
    {
        if (not_present(name, m_edges))
        {
            EdgeList el;
            el.m_name = name;
            el.m_labels = labels; 

            el.m_id = m_edges.size();
            el.m_offset = m_e_count;

            

            m_e_count += ; // ADD THIS;

            edges[el.m_id].emplace_back(std::move(el));
        }
    }
*/
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

                Weights my_wgts;
                my_wgts.m_name = wgts_name;
                my_wgts.m_weights = wgts;
                my_wgts.m_scaling = scaling;

                m_nodes[nl].m_wgts.emplace_back(std::move(my_wgts));
            }
        }
    }
/*
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
*/
    // -------------------------------- 

    template<class FAB>
    void Graph::addObj(const FabArray<FAB>& fab,
                       const std::string& node_name,
                       const std::vector<Real>& weights,
                       const std::string& wgts_name,
                       const Real scaling)
    {
        addNodeList(fab, node_name);
        addNodeWeight(node_name, wgts_name, weights, scaling);
    }

    void Graph::addObj(const DistributionMapping& dm,
                       const std::string& node_name,
                       const std::vector<std::string> labels,
                       const std::vector<Real>& weights,
                       const std::string& wgts_name,
                       const Real scaling)
    {
        addNodeList(dm, node_name, labels);
        addNodeWeight(node_name, wgts_name, weights, scaling);
    }

/*
    void Graph::addEdgeObj(const std::string& noe_name, 
                           const std::string& wgt_name)
    {
    }
*/
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

    void Graph::print(std::string filename)
    {
        std::ostringstream oss_r(nl.m_name + " = [", std::ios_base::ate);
        std::ostringstream oss_l(nl.m_name + "_labels = [", std::ios_base::ate);

        n_oss.precision(5); 

        for (int nid=0; nid<m_nodes.size(); ++nid)
        {
            const NodeList& nl = m_nodes[nid];

            oss_r << nl.m_name << " = [";
            for (int i=0; i<nl.m_ranks.size(); ++i)
            {
                oss_r << " " << std::to_string( );
            }             
        }

        amrex::PrintToFile file(filename);
        file << ;
    }
}

#endif
