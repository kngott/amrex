#include <AMReX_Print.H>
#include <AMReX_DotGraph.H>
#include <AMReX_ParallelDescriptor.H>
#include <string>

namespace amrex
{
    void dot_graph(const std::string& filename,
                   const DistributionMapping& dm,
                   const Vector<Real>& weights)
    {
        if (ParallelDescriptor::IOProcessor())
        {
            int nranks = ParallelDescriptor::NProcs();
            bool is_wgts = (weights.size() == dm.size());

            std::vector<std::string> subgraph(nranks, "");

            for (int i=0; i<dm.size(); ++i)
            {
                subgraph[dm[i]] += "    \"Box " + std::to_string(i);

                if (is_wgts) {
                    subgraph[dm[i]] += "\\n" + std::to_string(weights[i]);
                }

                subgraph[dm[i]] += "\"\n";
            }

            amrex::PrintToFile file("dot."+filename);
            file << "graph G {" << std::endl << std::endl;

            for (unsigned int i=0; i<subgraph.size(); ++i)
            {
                file << "  subgraph cluster_" << i <<  " {" << std::endl;
                file << subgraph[i];
                file << "    label = \"rank " << i << "\"" << std::endl;
                file << "  }" << std::endl << std::endl; 
            }
            file << "}";
        }
    }

    // ================================================================================
    // ================================================================================
    // ================================================================================

    void dot_graph_python(const std::string& filename,
                          const DistributionMapping& dm, 
                          const Vector<Real>& weights)
    {
        if (ParallelDescriptor::IOProcessor())
        {
            int nranks = ParallelDescriptor::NProcs();

            bool is_wgts = (weights.size() == dm.size());
            Real wgt_per_rank;
            std::vector<Real> subweight(nranks, 0);
            std::vector<std::string> subgraph(nranks, "");

            for (int i=0; i<dm.size(); ++i)
            {
                subgraph[dm[i]] += "    c.node('Box " + std::to_string(i);

                if (is_wgts) {
                    subgraph[dm[i]] += "\\n" + std::to_string(weights[i]);

                    wgt_per_rank += weights[i];
                    subweight[dm[i]] += weights[i];
                }

                subgraph[dm[i]] += "')\n";
            }

            wgt_per_rank /= nranks;

            amrex::PrintToFile file("dot."+filename);

            file << "dot = graphviz.Graph(engine='fdp')" << std::endl << std::endl;
            file << "dot.attr(label=r'\\n\\nAverage weight per rank: " 
                 << std::to_string(wgt_per_rank) << "')" << std::endl << std::endl;

            for (unsigned int i=0; i<subgraph.size(); ++i)
            {
                file << "with dot.subgraph(name='cluster_" << i << "') as c:" << std::endl;
                file << subgraph[i];
                file << "    c.attr(label='rank " << i; 
                if (is_wgts) {
                    file << ", wgt " << std::to_string(subweight[i]);
                }
                file <<  "')" << std::endl << std::endl;
            }
            file << "dot";
        }
    }
}
