#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "mylaunch.H"

using namespace amrex;

void main_main();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    Real strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    int n_cell, max_grid_size;
    Vector<int> ncell_vec(AMREX_SPACEDIM, 0);

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of 
        //   a square (or cubic) domain.
        n_cell = 0;
        pp.query("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Allow for 3D variability 
        for (auto& i : ncell_vec)
            { i = 0; }
        pp.queryarr("ncell_vec", ncell_vec);
    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        // If multi-dimensional domain not set in ncell_vec, use n_cell to set it.
        if (ncell_vec[0] == 0)
        {
            if (n_cell == 0)
            {
                amrex::Abort("Size of problem not set. Set either n_cell or ncell_vec in the inputs file.");
            }
            else
            {
                for(int i=0; i<AMREX_SPACEDIM; ++i)
                {
                    ncell_vec[i] = n_cell-1;
                }
            }        
        }

        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(ncell_vec[0]-1, ncell_vec[1]-1, ncell_vec[2]-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

        // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL(-1.0,-1.0,-1.0)},
                         {AMREX_D_DECL( 1.0, 1.0, 1.0)});

        // periodic in all direction
        Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

        // This defines a Geometry object
        geom.define(domain,real_box,CoordSys::cartesian,is_periodic);
    }

    // Nghost = number of ghost cells for each array 
    int Nghost = 0;
    
    // Ncomp = number of components for each array
    int Ncomp  = 1;
  
    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab mfab(ba, dm, Ncomp, Nghost);
    // ========================================

    for (MFIter mfi(mfab); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& arr = mfab.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            arr(i,j,k) = blockDim.x*blockIdx.x+threadIdx.x;
        }); 
    }

    amrex::Print() << "==================================" << std::endl;
    amrex::Print() << " Standard ParallelFor Launch " << std::endl;

    for (MFIter mfi(mfab); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& arr = mfab.array(mfi);

        amrex::ParallelForSMijk(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int tid = blockDim.x*blockIdx.x+threadIdx.x;
            printf("*** tid %i = %f = %i,%i,%i ***\n", tid, arr(i,j,k), i, j, k);
        }); 
    }

    amrex::Print() << std::endl << "==================================" << std::endl;
    amrex::Print() << " ParallelForjki Launch " << std::endl;

    for (MFIter mfi(mfab); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& arr = mfab.array(mfi);

        amrex::ParallelForSMjki(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int tid = blockDim.x*blockIdx.x+threadIdx.x;
            printf("*** tid %i = %f = %i,%i,%i ***\n", tid, arr(i,j,k), i, j, k);
        }); 
    }

    amrex::Print() << std::endl << "==================================" << std::endl;
    amrex::Print() << " ParallelForkij Launch " << std::endl;

    for (MFIter mfi(mfab); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& arr = mfab.array(mfi);

        amrex::ParallelForSMkij(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int tid = blockDim.x*blockIdx.x+threadIdx.x;
            printf("*** tid %i = %f = %i,%i,%i ***\n", tid, arr(i,j,k), i, j, k);
        }); 
    }

}
