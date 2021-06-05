#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>
//#include <AMReX_MultiFabUtil.H>

void main_main ();

typedef double Real;

// ================================================

Real MFdiff(const amrex::MultiFab& lhs, const amrex::MultiFab& rhs,
            int strt_comp, int num_comp, int nghost, const std::string name = "")
{
    amrex::MultiFab temp(rhs.boxArray(), rhs.DistributionMap(), rhs.nComp(), nghost);
    temp.ParallelCopy(lhs);
    temp.minus(rhs, strt_comp, num_comp, nghost);
/*
    if (name != "")
        { amrex::VisMF::Write(temp, std::string("pltfiles/" + name)); }
*/
    Real max_diff = 0;
    for (int i=0; i<num_comp; ++i)
    {
        Real max_i = std::abs(temp.max(i));
        max_diff = (max_diff > max_i) ? max_diff : max_i;
    }

    return max_diff; 
}

// ================================================

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}

void main_main ()
{

    BL_PROFILE("main");

    int ncell = 64;
    int ncomp = 1;
    int nboxes = 0;
    int nghost = 0;

    {
        amrex::ParmParse pp;
        pp.get("ncell", ncell);
        pp.query("nboxes", nboxes);
        pp.query("ncomp", ncomp);
        pp.query("nghost", nghost);
    }

    if (nboxes == 0)
        { nboxes = amrex::ParallelDescriptor::NProcs(); }

    amrex::MultiFab mf_src, mf_dst;
    amrex::IntVect ghosts(nghost);

// ***************************************************************
    // Build the Multifabs and Geometries.
    {
        amrex::Box domain(amrex::IntVect{0}, amrex::IntVect{ncell-1, ncell-1, nboxes*(ncell-1)});
        amrex::BoxArray ba(domain);
        ba.maxSize(ncell);

        amrex::Print() << "domain = " << domain << std::endl;
        amrex::Print() << "boxsize = " << ncell << std::endl;
        amrex::Print() << "nranks = " << nboxes << std::endl;
        amrex::Print() << "ncomp = " << ncomp << std::endl;
        amrex::Print() << "nghost = " << nghost << std::endl;

        amrex::DistributionMapping dm_src(ba);

        amrex::Vector<int> dst_map = dm_src.ProcessorMap();
        for (int& b : dst_map)
        {
           if (b != amrex::ParallelDescriptor::NProcs()-1)
               { b++; } 
           else 
               { b=0; }
        }

        amrex::DistributionMapping dm_dst(dst_map);

        Real val = 13.0;
        mf_src.define(ba, dm_src, ncomp, ghosts);
        mf_src.setVal(val++);

        mf_dst.define(ba, dm_dst, ncomp, ghosts);
        mf_dst.setVal(val++);
/*
        amrex::UtilCreateDirectoryDestructive("./pltfiles");

        amrex::VisMF::Write(mf_src, std::string("pltfiles/src_B"));
        amrex::VisMF::Write(mf_dst, std::string("pltfiles/dst_B"));
*/
        amrex::Print() << "dm = " << dm_src << std::endl;
        amrex::Vector<int> count(nboxes, 0);
        for (int& p: dst_map)
            { count[p]++; }
        for (int i=0; i<count.size(); ++i)
            { amrex::Print() << "count[" << i << "]: " << count[i] << std::endl; }

    }

    {   
        BL_PROFILE("**** Test - 1st");
        mf_dst.ParallelCopy(mf_src);
    }
/*
    {
        BL_PROFILE("**** Test - 2nd");
        CPC c_pattern(mf_dst, ghosts, mf_src, ghosts, amrex::Periodicity::NonPeriodic());

        ParallelCopy(mf_dst, mf_src, 0, 0, ncomp,
                     ghosts, ghosts, amrex::Periodicity::NonPeriodic(),
                     COPY, c_pattern);
    }

    amrex::VisMF::Write(mf_src, std::string("pltfiles/src_after"));
    amrex::VisMF::Write(mf_dst, std::string("pltfiles/dst_after"));
*/

    amrex::Print() << "Error in old PC: " 
                   << MFdiff(mf_src, mf_dst, 0, ncomp, nghost) << std::endl;
}
