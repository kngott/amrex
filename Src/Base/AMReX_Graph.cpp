#include <AMReX_Graph.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Config.H>
#include <AMReX_FabArray.H>
#include <AMReX_LayoutData.H>

#include <numeric>

namespace amrex {

void Graph::addFab(const FabArrayBase& fab,
                   const std::string& name,
                   const size_t data_size)
{
    if (is_present(name, m_nodes)) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.addFab() called for a fab already named in the graph -- "
                       << name << ". Returning." << std::endl;
        return;
    }

    addNodeList(name, fab, data_size);
}

// --------------------------------

void Graph::addNodeList(const std::string& name,
                        const FabArrayBase& fab,
                        const std::size_t& item_size)
{
    if (is_present(name, m_nodes)) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.addNodeList() called for a nodelist already named in the graph -- "
                       << name << ". Returning." << std::endl;
        return;
    }

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

void Graph::addEdgeList(const std::string& name,
                        const std::string& from_name,
                        const std::string& to_name,
                        const double scaling,
                        const FabArrayBase::CommMetaData& comm_data,
                        const int ncomp)
{
    // label => snd box
    // weight => bytes sent

    if (is_present(name, m_edges)) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.addEdgeList() called for a edgelist already named in the graph -- "
                       << name << ". Returning." << std::endl;
        return;
    }

    EdgeList el;
    el.m_name = name;
    el.m_id = m_edges.size();
    el.m_offset = m_e_count;
    el.m_mynodes.first = from_name;
    el.m_mynodes.second = to_name;

    if (not_present(from_name, m_nodes) || not_present(to_name, m_nodes)) {
        amrex::Abort("Node lists" + from_name + " or " + to_name + " are not present in the graph.");
    }

    // Don't do recvs, so comms aren't duplicated.
    const int N_locs = comm_data.m_LocTags->size();
    int N_snds = 0;
    for (const auto& cctc : *comm_data.m_SndTags) { N_snds += cctc.second.size(); }

//    const int N_snds = comm_data.m_SndTags->size();
    el.m_size = (N_locs + N_snds);
    m_e_count += el.m_size;
    el.m_from.reserve(el.m_size);
    el.m_to.reserve(el.m_size);
    el.m_labels.reserve(el.m_size);

    std::vector<double> weights;
    weights.reserve(el.m_size);

    amrex::Print() << "size = " << (N_locs +N_snds) << ": "
                   << N_locs << " " << N_snds << std::endl;

    int from_id = get_index(from_name, m_nodes);
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
        amrex::Print() << "LOC" << std::endl;
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
            amrex::Print() << "SND" << std::endl;
        }
    }

    m_edges.emplace_back(std::move(el));
    addEdgeWeight(name, "bytes", weights, scaling);
}

// --------------------------------

void Graph::appendEdgeList(const std::string& name,
                           const std::string& from_name,
                           const std::string& to_name,
                           const double scaling,
                           const FabArrayBase::CommMetaData& comm_data,
                           const int ncomp)
{
    int el_index = get_index(name, m_edges);

    if (el_index == -1) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.appendEdgeList() called for a edgelist not in the graph -- "
                       << name << ". Returning." << std::endl;
        return;
    }

    EdgeList& el = m_edges[el_index];

    if ((from_name != el.m_mynodes.first) || (to_name != el.m_mynodes.second)) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.appendEdgeList() called on a edgelist with non-matching to and from nodelists -- "
                       << from_name << " -> " << to_name << ". Returning." << std::endl;
        return;
    }

    // Don't do recvs, so comms aren't duplicated.
    const int N_locs = comm_data.m_LocTags->size();
    int N_snds = 0;
    for (const auto& cctc : *comm_data.m_SndTags) { N_snds += cctc.second.size(); }

    el.m_size += (N_locs + N_snds);
    m_e_count += (N_locs + N_snds);
    el.m_from.reserve(el.m_size);
    el.m_to.reserve(el.m_size);
    el.m_labels.reserve(el.m_size);

    amrex::Print() << "size = " << (N_locs +N_snds) << ": "
                   << N_locs << " " << N_snds << std::endl;

    // Update this edgelist and all with higher indexes.
    for (unsigned int i=el_index; i<m_edges.size(); ++i) {
        m_edges[el_index].m_offset += (N_locs + N_snds);
    }

    std::vector<double> weights;
    weights.reserve(N_locs + N_snds);

    int from_id = get_index(from_name, m_nodes);
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

    //m_edges.emplace_back(std::move(el));
    appendEdgeWeight(name, "bytes", weights, scaling);
}

// --------------------------------

void Graph::addNodeWeight(const std::string& node_name,
                          const std::string& wgts_name,
                          const std::vector<double>& wgts,
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
    new_wgt.m_weights = wgts;
    new_wgt.m_scaling[0] = scaling;
    new_wgt.m_local = local;

    m_nodes[nl].m_wgts.emplace_back(std::move(new_wgt));

    m_nwgts.push_back(wgts_name);
}

void Graph::addEdgeWeight(const std::string& edge_name,
                          const std::string& wgts_name,
                          const std::vector<double>& wgts,
                          const double scaling,
                          const bool local)
{
    int el = get_index(edge_name, m_edges);

    if (el == -1) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.addEdgeWeight() called for a edgelist not in the graph -- "
                       << edge_name << ". Returning." << std::endl;
        return;
    }
    if (is_present(wgts_name, m_edges[el].m_wgts)) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.addEdgeWeight() called for a weight already named in the graph -- "
                       << wgts_name << ". Returning." << std::endl;
        return;
    }

    // Check length of weights is correct
    AMREX_ASSERT(wgts.size() == m_edges[el].m_from.size());

    Weight new_wgt;
    new_wgt.m_name = wgts_name;
    new_wgt.m_weights = wgts;
    new_wgt.m_scaling[0] = scaling;
    new_wgt.m_local = local;

    m_edges[el].m_wgts.emplace_back(std::move(new_wgt));
/*
    if (wgts_name == "bytes") {
        m_ewgts.push_back(edge_name + "_" + wgts_name);
    } else {
        m_ewgts.push_back(wgts_name);
    }
*/
}

// --------------------------------

void Graph::appendEdgeWeight(const std::string& edge_name,
                             const std::string& wgts_name,
                             const std::vector<double>& wgts,
                             const double scaling,
                             const bool local)
{
    int el = get_index(edge_name, m_edges);

    if (el == -1) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.appendEdgeWeight() called for a edgelist not in the graph -- "
                       << edge_name << ". Returning." << std::endl;
        return;
    }

    int w_idx = get_index(wgts_name, m_edges[el].m_wgts);

    if (w_idx == -1) {
        amrex::Print() << " **** WARNING: "
                       << " Graph.appendEdgeWeight() called for a weight not in the graph -- "
                       << wgts_name << ". Returning." << std::endl;

        for (unsigned int i=0; i<m_edges[el].m_wgts.size(); ++i) {
            amrex::Print() << "NAME: " << m_edges[el].m_wgts[i].m_name << std::endl;
        }
        return;
    }

    // Check length of weights is correct
    AMREX_ASSERT( m_edges[el].m_from.size() == (wgts.size() + m_wgts[el].m_ewgts[w_idx]) );

    Weight& new_wgt = m_edges[el].m_wgts[w_idx];
//    new_wgt.m_name = wgts_name;
    new_wgt.m_weights.insert(new_wgt.m_weights.end(), wgts.begin(), wgts.end());
    new_wgt.m_scaling[0] = scaling;        // Update these in append
    new_wgt.m_local = local;               // Update these in append

//    m_edges[el].m_wgts.emplace_back(std::move(new_wgt));

    if (wgts_name == "bytes") {
        m_ewgts.push_back(edge_name + "_" + wgts_name);
    } else {
        m_ewgts.push_back(wgts_name);
    }
}

// --------------------------------

void Graph::clear()
{
    // Leave m_rank, for matching re-use.

    m_assembled = false;
    m_n_count = 0;
    m_e_count = 0;
    m_nwgts.clear();
    m_ewgts.clear();
    m_nodes.clear();
    m_edges.clear();
}

Graph
Graph::assemble()
{
    // If this is assembled, return a copy.
    if (m_assembled) { return amrex::Graph() = *this; }

    // Make a copy and add full data from other ranks to the full_graph on m_rank.

    Graph full_graph;
    bool is_writer = (ParallelDescriptor::MyProc() == m_rank);

    if (is_writer) {
        full_graph.m_assembled = true;

        full_graph.m_rank = m_rank;
        full_graph.m_n_count = m_n_count;
        full_graph.m_e_count = 0;          // Local data, so counted here.
        full_graph.m_nwgts = m_nwgts;
        full_graph.m_ewgts = m_ewgts;
    }

    const int n_ranks = ParallelDescriptor::NProcs();

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

            if ( wgt.m_weights.size() < (unsigned int) (fab.size()) )
            {
                for (MFIter mfi(fab); mfi.isValid(); ++mfi)
                {
                    // Assuming tiling is off.
                    lod[mfi] = wgt.m_weights[mfi.LocalIndex()];
                }

                Vector<double> collection(fab.size(), 0.0);
                ParallelDescriptor::GatherLayoutDataToVector<double>(lod, collection, m_rank);

                if (is_writer) { full_wgt.m_weights = collection; }
            }

            // For all weights, collect scaling.
            double my_value = full_wgt.m_scaling[0];

            full_wgt.m_scaling.resize(n_ranks);
            full_wgt.m_scaling = ParallelDescriptor::Gather<double>(my_value, m_rank);
        }
    }

    // ====================================

    int offset_count = 0;

    // Edges: aggregate all information.
    // Need to protect from aggregating over and over and over again.
    for (unsigned int i=0; i<m_edges.size(); ++i)
    {
        full_graph.m_edges.push_back(m_edges[i]);
        EdgeList& el = full_graph.m_edges[i];

        // ---------------------------
        // Send # of edges & size of labels for all of them

        // ... Edges ...
        int n_local_edges = el.m_from.size();
        std::vector<int> edge_count = ParallelDescriptor::Gather<int>(n_local_edges, m_rank);
        std::vector<int> disp(n_ranks, 0);
        int n_total_edges = 0;

        if (is_writer) {
            n_total_edges = amrex::Scan::ExclusiveSum(n_ranks, edge_count.data(), disp.data());
            el.m_size = n_total_edges;
            el.m_offset = offset_count;
            offset_count += n_total_edges;
            full_graph.m_e_count += n_total_edges;
        }

        // ... Labels ...
        int n_local_chars = 0;
        std::vector<int> char_count(n_local_edges, 0);
        for (int j=0; j<n_local_edges; ++j) {
            char_count[j] = el.m_labels[j].size();
            n_local_chars += el.m_labels[j].size();
        }

        std::vector<int> char_sums = ParallelDescriptor::Gather<int>(n_local_chars, m_rank);
        std::vector<int> label_disp (n_ranks, 0);
        int n_global_chars = 0;

        if (is_writer) {
            n_global_chars = amrex::Scan::ExclusiveSum(n_ranks, char_sums.data(),
                                                       label_disp.data());
        }

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

            ParallelDescriptor::Gatherv<int>(el.m_from.data(), n_local_edges,
                                             all_from.data(), edge_count, disp, m_rank);
            ParallelDescriptor::Gatherv<int>(el.m_to.data(), n_local_edges,
                                             all_to.data(), edge_count, disp, m_rank);

            if (is_writer) {
                el.m_from = std::move(all_from);
                el.m_to = std::move(all_to);
            }
        }

        // .....................
        // Labels
        {
            char* local_clabel = static_cast<char*> (amrex::The_Cpu_Arena()->alloc(sizeof(char)*n_local_chars));

            char* global_clabel = nullptr;
            if (is_writer) {
                global_clabel = static_cast<char*> (amrex::The_Cpu_Arena()->alloc(sizeof(char)*n_global_chars));
            }

            // Do it manually. Better way?
            long index = -1;
            for (unsigned int j=0; j<el.m_labels.size(); ++j)
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
            if (is_writer) {
                char* c = global_clabel;
                for (int j=0; j<n_total_edges; ++j) {
                    unpack_labels[j].assign(c, label_sizes[j]);
                    c += label_sizes[j];
                }

                el.m_labels = std::move(unpack_labels);

                amrex::The_Cpu_Arena()->free(global_clabel);
            }

            amrex::The_Cpu_Arena()->free(local_clabel);
        }

        // ......................
        // Weights and scalings

        for (unsigned int w=0; w<el.m_wgts.size(); ++w)
        {
            const Weight& wgt = el.m_wgts[w];
            Weight& full_wgt = full_graph.m_edges[i].m_wgts[w];

            Vector<double> collection(n_total_edges, 0.0);

            ParallelDescriptor::Gatherv<double>(wgt.m_weights.data(), n_local_edges,
                                                collection.data(), edge_count, disp, m_rank);
            if (is_writer) {
                full_wgt.m_weights = std::move(collection);
            }

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
    Graph fullg = this->assemble();
    fullg.print_doit(filename, wgt_precision);
}

void Graph::print_table(const std::string& filename,
                  const int wgt_precision,
                  const bool /*replace_file*/)    // std::rename ?
{
    Graph fullg = this->assemble();
    fullg.print_table_doit(filename, wgt_precision);
}

void Graph::print_doit(const std::string& filename,
                       const int wgt_precision,
                       const bool /*replace_file*/)    // std::rename ?
{
    if (ParallelDescriptor::MyProc() != m_rank)  { return; }

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

            file << oss_w.str() << std::endl << std::endl;
        }
    }

//    file << std::endl << std::endl;

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

void Graph::print_table_doit(const std::string& dirname,
                             const int wgt_precision,
                             const bool /*replace_file*/)    // std::rename ?
{
    if (ParallelDescriptor::MyProc() != m_rank)  { return; }

    std::string fulldirname = std::string("graphs/") + dirname;
    amrex::UtilCreateCleanDirectory(fulldirname, false);

    /*
        Tables:

        Name = directory name
        Graph/name/files

        edges.txt        --   edge# from(box#) to(box#) label ewgt1 ewgt2 ewgt3
        edgelists.txt    --   edge_name node_to node_from size start(row#) end(row#)
        edgescaling.txt  --   rank ewgt1 ewgt2 ewgt3

        nodes.txt        --   box#(row#) rank label nwgt1 nwgt2 nwgt3
        nodelists.txt    --   fab_name size start(row#) end(row#)
        nodescaling.txt  --   rank nwgt1 nwgt2 nwgt3
    */

    // Nodes
    {
        std::ostringstream  n_ss(std::ios_base::ate);
        std::ostringstream nl_ss(std::ios_base::ate);
        std::ostringstream ns_ss(std::ios_base::ate);
        std::ostringstream nw_ss(std::ios_base::ate);

        long node_id = 0;

        std::vector< std::vector<double> const* > smap(m_nwgts.size(), nullptr);

        // Create table headers
        if (m_nwgts.size() > 0) { nw_ss << "\"" << m_nwgts[0] << "\""; }
        for (unsigned int w=1; w<m_nwgts.size(); ++w) { nw_ss << " \"" << m_nwgts[w] << "\""; }
        n_ss << "box-# rank label " << nw_ss.str() << std::endl;
        nl_ss << "name type-size index size start-id end-id " << std::endl;
        ns_ss << "rank " << nw_ss.str() << std::endl;

        for (unsigned int nid=0; nid<m_nodes.size(); ++nid)
        {
            const NodeList& nl = m_nodes[nid];

            nl_ss << "\"" << nl.m_name
                  << "\" " << std::to_string(nl.m_bytes_per_item)
                  << " \"" << nl.m_fab.ixType()
                  << "\" " << std::to_string(nl.m_size)
                  << " "   << std::to_string(nl.m_offset)
                  << " "   << std::to_string(nl.m_offset+nl.m_size-1) << "\n";

            std::vector<int> wgtmap(m_nwgts.size(), -1);;
            for (unsigned int w=0; w<m_nwgts.size(); ++w) {
                const int idx = get_index(m_nwgts[w], nl.m_wgts);
                if (idx != -1) {
                    wgtmap[w] = idx;
                    smap[w] = &(nl.m_wgts[idx].m_scaling);
                }
            }

            for (int i=0; i<nl.m_size; ++i) {
                const int rank = nl.m_fab.DistributionMap()[i];
                const Box& bx = nl.m_fab.boxArray()[i];

                // to::string to prevent any precision-based round off.
                n_ss << node_id << " " << std::to_string(rank)
                     << " \"" << bx.smallEnd() << " " << bx.bigEnd() << "\" ";

                n_ss.precision(wgt_precision);

                for (const auto idx : wgtmap) {
                    if (idx != -1) {
                        n_ss << " " << nl.m_wgts[idx].m_weights[i];
                    } else {
                        n_ss << " null";
                    }
                }
                n_ss << std::endl;
                node_id++;
            }
        }

        ns_ss.precision(wgt_precision);
        for (int n=0; n<ParallelDescriptor::NProcs(); ++n) {
            ns_ss << std::to_string(n);
            for (unsigned int s=0; s<smap.size(); ++s) {
                ns_ss << " " << (*(smap[s]))[n];
            }
            ns_ss << "\n";
        }

        amrex::PrintToFile n_file(fulldirname + std::string("/nodes.txt"));
        amrex::PrintToFile nl_file(fulldirname + std::string("/nodelists.txt"));
        amrex::PrintToFile ns_file(fulldirname + std::string("/nodescaling.txt"));

        n_file << n_ss.str();
        nl_file << nl_ss.str();
        ns_file << ns_ss.str();
    }

    // Edges.
    {
        std::ostringstream  e_ss(std::ios_base::ate);
        std::ostringstream el_ss(std::ios_base::ate);
        std::ostringstream es_ss(std::ios_base::ate);
        std::ostringstream ew_ss(std::ios_base::ate);
        long edge_id = 0;

        std::vector< std::vector<double> const* > smap(m_ewgts.size(), nullptr);

        // For table headers
        if (m_ewgts.size() > 0) { ew_ss << "\"" << m_ewgts[0] << "\""; }
        for (unsigned int w=1; w<m_ewgts.size(); ++w) { ew_ss << " \"" << m_ewgts[w] << "\""; }
        e_ss << "edge-# from-box to-box label " << ew_ss.str() << std::endl;
        el_ss << "name nl_to nl_from size start-id end-id" << std::endl;
        es_ss << "rank " << ew_ss.str() << std::endl;


        for (unsigned int eid=0; eid<m_edges.size(); ++eid)
        {
            const EdgeList& el = m_edges[eid];

            el_ss << "\""  << el.m_name << "\" "
                  << "\"" << el.m_mynodes.first << "\" "
                  << "\"" << el.m_mynodes.second << "\" "
                  << std::to_string(el.m_size) << " " << std::to_string(el.m_offset) << " "
                  << std::to_string(el.m_offset+el.m_size-1) << "\n";

            int from_idx = get_index(el.m_mynodes.first, m_nodes);
            int to_idx = get_index(el.m_mynodes.second, m_nodes);

            int from_offset = m_nodes[from_idx].m_offset;
            int to_offset = m_nodes[to_idx].m_offset;

            std::vector<int> wgtmap(m_ewgts.size(), -1);
            const int idx_b = get_index("bytes", el.m_wgts);

            for (unsigned int w=0; w<m_ewgts.size(); ++w) {
                const int idx = get_index(m_ewgts[w], el.m_wgts);

                if (idx != -1) {
                    wgtmap[w] = idx;
                    smap[w] = &(el.m_wgts[idx].m_scaling);
                }
                else if (m_ewgts[w] == el.m_name+"_bytes") {
                    wgtmap[w] = idx_b;
                    smap[w] = &(el.m_wgts[idx_b].m_scaling);
                }
            }

            for (int i=0; i<el.m_size; ++i) {
                int global_from = el.m_from[i] + from_offset;
                int global_to = el.m_to[i] + to_offset;

                // to::string to prevent any precision-based round off.
                e_ss << edge_id << " " << std::to_string(global_from)
                                << " " << std::to_string(global_to)
                                << " " << el.m_labels[i];

                e_ss.precision(wgt_precision);

                for (const auto idx : wgtmap) {
                    if (idx != -1) {
                        e_ss << " " << el.m_wgts[idx].m_weights[i];
                    } else {
                        e_ss << " null";
                    }
                }
                e_ss << "\n";
                edge_id++;
            }
        }

        es_ss.precision(wgt_precision);
        for (int n=0; n<ParallelDescriptor::NProcs(); ++n) {
            es_ss << std::to_string(n);
            for (unsigned int s=0; s<smap.size(); ++s) {
                es_ss << " " << (*(smap[s]))[n];
            }
            es_ss << "\n";
        }

        amrex::PrintToFile e_file(fulldirname + std::string("/edges.txt"));
        amrex::PrintToFile el_file(fulldirname + std::string("/edgelists.txt"));
        amrex::PrintToFile es_file(fulldirname + std::string("/edgescaling.txt"));

        e_file << e_ss.str();
        el_file << el_ss.str();
        es_file << es_ss.str();
    }
}

}  // namespace amrex
