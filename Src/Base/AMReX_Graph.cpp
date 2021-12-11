
#include <AMReX_Graph.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>
#include <AMReX_LayoutData.H>

#include <numeric>

namespace amrex {

void Graph::addNodeList(const std::string& name,
                        const FabArrayBase& fab,
                        const std::size_t& item_size)
{
    if (not_present(name, m_nodes))
    {
        NodeList nl;
        nl.m_name = name;
        nl.m_id = m_nodes.size();
        nl.m_offset = m_n_count;
        nl.m_size = fab.DistributionMap().size();
        m_n_count += nl.m_size;

        nl.m_fab = fab;
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
        el.m_from.reserve(el.m_size);
        el.m_to.reserve(el.m_size);
        el.m_labels.reserve(el.m_size);

        std::vector<double> weights;
        weights.reserve(el.m_size);

        int from_id = get_index(from_name, m_nodes);
//        int to_id = get_index(to_name, m_nodes);

//        int offset_src = m_nodes[from_id].m_offset;
//        int offset_dst = m_nodes[to_id].m_offset;
        int type_size = m_nodes[from_id].m_bytes_per_item;

        const auto& LocTags = comm_data.m_LocTags;
        const auto& SndTags = comm_data.m_SndTags;

        // Combination of multiple ranges would make this much nicer. :)
        // Or, subfunction/lambda function to do this once.
        for (const FabArrayBase::CopyComTag& cct : *LocTags) {
            std::ostringstream oss("\"", std::ios_base::ate);
            const Box& bx = cct.sbox;
            oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

            el.m_from.emplace_back(cct.srcIndex);
            el.m_to.emplace_back(cct.dstIndex);
            el.m_labels.emplace_back(oss.str());
            weights.emplace_back(bx.numPts()*type_size*ncomp);
        }

        for (const auto& kv: *SndTags) {
            for (const auto& cct : kv.second) {
                std::ostringstream oss("\"", std::ios_base::ate);
                const Box& bx = cct.sbox;
                oss << bx.smallEnd() << " " << bx.bigEnd() << "\"";

                el.m_from.emplace_back(cct.srcIndex);
                el.m_to.emplace_back(cct.dstIndex);
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
                          const std::vector<double>& wgts,
                          const double scaling,
                          const bool local)
{
    int nl = get_index(node_name, m_nodes);

    if (nl != -1)
    {
        if (not_present(wgts_name, m_nodes[nl].m_wgts))
        {
            // Check length of weights is correct
//            AMREX_ASSERT(wgts.size() == m_nodes[nl].m_ranks.size() );

            Weight new_wgt;
            new_wgt.m_name = wgts_name;
            new_wgt.m_weights = wgts;
            new_wgt.m_scaling[0] = scaling;
            new_wgt.m_local = local;

            m_nodes[nl].m_wgts.emplace_back(std::move(new_wgt));
        }
    }
}

void Graph::addEdgeWeight(const std::string& edge_name,
                          const std::string& wgts_name,
                          const std::vector<double>& wgts,
                          const double scaling,
                          const bool local)
{
    int el = get_index(edge_name, m_edges);

    if (el != -1)
    {
        if (not_present(wgts_name, m_edges[el].m_wgts))
        {
            // Check length of weights is correct
            AMREX_ASSERT(wgts.size() == m_edges[el].m_from.size());

            Weight new_wgt;
            new_wgt.m_name = wgts_name;
            new_wgt.m_weights = wgts;
            new_wgt.m_scaling[0] = scaling;
            new_wgt.m_local = local;

            m_edges[el].m_wgts.emplace_back(std::move(new_wgt));
        }
    }
}

// --------------------------------

void Graph::clear()
{
    m_n_count = 0;
    m_e_count = 0;
    m_nodes.clear();
    m_edges.clear();
}

Graph Graph::assemble()
{
    // If un-assembled, else return copy of self?
    //   -- Need to add accessor to assembled.
    // (If local weights ever added or updated,
    //  or if new node or edge list added to an assembled graph).
    // Graph started "assembled" -- blank is assembled.

    Graph full_graph;

    // Set assembled to false on non-host ranks?
//    if (ParallelDescriptor::MyProc() == m_rank) {}

    // Nodes: collect local weights.
    for (unsigned int i=0; i<m_nodes.size(); ++i)
    {
        full_graph.m_nodes.push_back(m_nodes[i]);
        const NodeList& nl = full_graph.m_nodes[i];

        const FabArrayBase& fab = nl.m_fab;
        LayoutData<double> lod(fab.boxArray(), fab.DistributionMap());

        for (unsigned int w=0; w<nl.m_wgts.size() ; ++w)
        {
            const Weight& wgt = nl.m_wgts[w];
            Weight& full_wgt = full_graph.m_nodes[i].m_wgts[w];

            // If non-global, update it.
            if ( nl.m_wgts[w].m_weights.size() < (unsigned int) (fab.size()) )
            {
                for (MFIter mfi(fab); mfi.isValid(); ++mfi)
                {
                    // I believe this is correct, assuming tiling is off.
                    lod[mfi] = wgt.m_weights[mfi.LocalIndex()];
                }

                Vector<double> collection(fab.size(), 0.0);
                ParallelDescriptor::GatherLayoutDataToVector<double>(lod, collection, m_rank);

                full_wgt.m_weights = collection;
            }

            // For all weights, collect scaling.
            double my_value = full_wgt.m_scaling[0];

            full_wgt.m_scaling.resize(ParallelDescriptor::NProcs());
            full_wgt.m_scaling = ParallelDescriptor::Gather<double>(my_value, m_rank);
        }
    }

    // ====================================
    // ====================================

    // Edges: aggregate all information.
    // Need to protect from aggregating over and over and over again.
    for (unsigned int i=0; i<m_edges.size(); ++i)
    {
        full_graph.m_edges.push_back(m_edges[i]);
        EdgeList& el = full_graph.m_edges[i];
        int n_ranks = ParallelDescriptor::NProcs();

        // ---------------------------
        // Send # of edges & size of labels for all of them

        int n_local_edges = el.m_from.size();
        std::vector<int> edge_count = ParallelDescriptor::Gather<int>(n_local_edges, m_rank);
        std::vector<int> disp(n_ranks, 0);
        int n_total_edges = amrex::Scan::ExclusiveSum(n_ranks, edge_count.data(), disp.data());

        int n_local_chars = 0;
        std::vector<int> char_count(n_local_edges, 0);
        for (int j=0; j<n_local_edges; ++j) {
            char_count[j] = el.m_labels[j].size();
            n_local_chars += el.m_labels[j].size();
        }
        std::vector<int> char_sums = ParallelDescriptor::Gather<int>(n_local_chars, m_rank);
        std::vector<int> label_disp (n_ranks, 0);

        // m_rank only?
        int n_global_chars = amrex::Scan::ExclusiveSum(n_ranks, char_sums.data(),
                                                         label_disp.data());


        std::vector<int> label_sizes(n_total_edges, 0);
        ParallelDescriptor::Gatherv<int>(char_count.data(), n_local_edges,
                                         label_sizes.data(), edge_count, disp, m_rank);

        // ---------------------------
        // Send pairs (2*E), labels (sum sizes), weights (num weights * E), and scalings (E)

        // .....................
        // Connection Pairs
        {
            std::vector<int> all_from(n_total_edges, -1);
            std::vector<int> all_to(n_total_edges, -1);

            el.m_from.resize(n_total_edges);
            el.m_to.resize(n_total_edges);

            ParallelDescriptor::Gatherv<int>(el.m_from.data(), n_local_edges,
                                             all_from.data(), edge_count, disp, m_rank);
            ParallelDescriptor::Gatherv<int>(el.m_to.data(), n_local_edges,
                                             all_to.data(), edge_count, disp, m_rank);

            // Can comm in place? Or keep like this to ensure ordering.
            el.m_from = std::move(all_from);
            el.m_to = std::move(all_to);
        }

        // .....................
        // Labels
        {
            char* local_clabel = static_cast<char*> (amrex::The_Cpu_Arena()->alloc(sizeof(char)*n_local_chars));

            char* global_clabel = nullptr;
            if (ParallelDescriptor::MyProc() == m_rank) {
                global_clabel = static_cast<char*> (amrex::The_Cpu_Arena()->alloc(sizeof(char)*n_global_chars));
            }

            // Do it manually. Better way?
            unsigned long index = -1;
            for (int j=0; j<el.m_labels.size(); ++j)
            {
                for (char k : el.m_labels[j]) {
                    local_clabel[++index] = k;
                }
            }

            ParallelDescriptor::Gatherv<char>(local_clabel, n_local_chars,
                                              global_clabel, char_sums, label_disp, m_rank);

            std::vector<std::string> unpack_labels(n_total_edges);

            // Unpack based on Gather-ed sizes. Kept like this until packing method is
            //   determined. Other option: null terminated.
            if (ParallelDescriptor::MyProc() == m_rank) {
                char* c = global_clabel;
                for (int j=0; j<n_total_edges; ++j) {
                    unpack_labels[j].assign(c, label_sizes[j]);
                    c += label_sizes[j];
                }
            }

            el.m_labels = std::move(unpack_labels);

            amrex::The_Cpu_Arena()->free(local_clabel);
            if (ParallelDescriptor::MyProc() == m_rank) {
                amrex::The_Cpu_Arena()->free(global_clabel);
            }
        }

        // ......................
        // Weights and scalings

        for (unsigned int w=0; w<el.m_wgts.size(); ++w)
        {
//          collect weights & scalings -- node copy?
            const Weight& wgt = el.m_wgts[w];
            Weight& full_wgt = full_graph.m_edges[i].m_wgts[w];

            Vector<double> collection(n_total_edges, 0.0);

            ParallelDescriptor::Gatherv<double>(wgt.m_weights.data(), n_local_edges,
                                                collection.data(), edge_count, disp, m_rank);

            full_wgt.m_weights = std::move(collection);

            // For all weights, collect scaling.
            double my_value = full_wgt.m_scaling[0];

            full_wgt.m_scaling.resize(ParallelDescriptor::NProcs());
            full_wgt.m_scaling = ParallelDescriptor::Gather<double>(my_value, m_rank);
        }
    }

    return full_graph;
}

void Graph::print(const std::string& filename,
                  const int wgt_precision,
                  const bool /*replace_file*/)    // std::rename ?
{
    Graph assembled = this->assemble();
    full_graph.print_doit(filename, wgt_precision);
}


void Graph::print_doit(const std::string& filename,
                       const int wgt_precision,
                       const bool /*replace_file*/)    // std::rename ?
{

    // ............ ADD: If graph is not already assembled.

    if ( !ParallelDescriptor::IOProcessor() )  { return; }

    amrex::PrintToFile file(filename);

    /*
        For NodeLists:

        name = [...]
        name_labels = [...]
        name_weight = [...]
        name_weight_scaling =
    */

    for (unsigned int nid=0; nid<m_nodes.size(); ++nid)
    {
        const NodeList& nl = m_nodes[nid];

        std::ostringstream oss_r(std::ios_base::ate);
        std::ostringstream oss_l(std::ios_base::ate);

        oss_r << nl.m_name << " = [";
        oss_l << nl.m_name << "_labels = [";

        for (int i=0; i<nl.m_size; ++i) {
            const Box& bx = nl.m_fab.boxArray()[i];

            oss_r << " " << std::to_string(nl.m_fab.DistributionMap()[i]); // To ensure no round-off.
            oss_l << bx.smallEnd() << "-" << bx.bigEnd() << " ";
        }
        oss_r << "]\n";
        oss_l << "]\n";

        file << oss_r.str() << std::endl << oss_l.str() << std::endl;


        for (unsigned int w=0; w<nl.m_wgts.size(); ++w)
        {
            const Weight& wt = nl.m_wgts[w];
            std::string this_name = nl.m_name + "_" + wt.m_name;

            std::ostringstream oss_w(std::ios_base::ate);
            oss_w.precision(wgt_precision);

            oss_w << this_name << " = [";
            for (unsigned int i=0; i<wt.m_weights.size(); ++i) {
                oss_w << " " << wt.m_weights[i];
            }

            oss_w << "]\n" << this_name << "_scaling = [";
            for (unsigned int i=0; i<wt.m_scaling.size(); ++i) {
                oss_w << " " << wt.m_scaling[i];
            }
            oss_w << "]\n";

            file << oss_w.str() << std::endl;
        }
    }

    file << std::endl << std::endl;

    /*
       For EdgeLists:
       name = [  ]
       name_src = from
       name_dst = to
       name_labels = [ ]
       name_weight = [ ]
       name_weight_scaling =
    */

    for (unsigned int eid=0; eid<m_edges.size(); ++eid)
    {
        if (eid==0) {
            file << std::endl << std::endl;
        }

        const EdgeList& el = m_edges[eid];

        std::ostringstream oss_e(std::ios_base::ate);
        std::ostringstream oss_l(std::ios_base::ate);

        oss_e << el.m_name << " = [";
        oss_l << el.m_name << "_labels = [";

        for (unsigned int i=0; i<el.m_from.size(); ++i) {
            oss_e << " (" << std::to_string(el.m_from[i]) << ","
                          << std::to_string(el.m_to[i]) << ")";    // To ensure no round-off.
            oss_l << " " << el.m_labels[i];
        }
        oss_e << "]\n";
        oss_l << "]\n";

        file << oss_e.str() << std::endl
             << el.m_name << "_src = " << el.m_mynodes.first << std::endl
             << el.m_name << "_dst = " << el.m_mynodes.second << std::endl
             << oss_l.str() << std::endl;

        for (unsigned int w=0; w<el.m_wgts.size(); ++w)
        {
            const Weight& wt = el.m_wgts[w];
            std::string this_name = el.m_name + "_" + wt.m_name;

            std::ostringstream oss_w(std::ios_base::ate);
            oss_w.precision(wgt_precision);

            oss_w << this_name << " = [";
            for (unsigned int i=0; i<el.m_from.size(); ++i) {
                oss_w << " " << wt.m_weights[i];
            }

            oss_w << "]\n" << this_name << "_scaling = [";
            for (unsigned int i=0; i<wt.m_scaling.size(); ++i) {
                oss_w << " " << wt.m_scaling[i];
            }
            oss_w << "]\n";

            file << oss_w.str() << std::endl;
        }
    }
}

}  // namespace amrex
