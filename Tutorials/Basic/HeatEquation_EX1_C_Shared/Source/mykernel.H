#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_phi (int i, int j, int k,
               amrex::Array4<amrex::Real> const& phi,
               GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    amrex::Real x = prob_lo[0] + (i+0.5) * dx[0];
    amrex::Real y = prob_lo[1] + (j+0.5) * dx[1];
#if (AMREX_SPACEDIM > 2)
    amrex::Real z = prob_lo[2] + (k+0.5) * dx[2];
#else
    amrex::Real z = 0.;
#endif
    amrex::Real r2 = ((x-0.25)*(x-0.25)+(y-0.25)*(y-0.25)+(z-0.25)*(z-0.25))/0.01;
    phi(i,j,k) = 1. + std::exp(-r2);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dxinv)
{
//    fluxx(i,j,k) = (phi(i,j,k)-phi(i-1,j,k)) * dxinv;
//    fluxx(i,j,k) = (phi(i+1,j,k)-2*phi(i,j,k)+phi(i-1,j,k)) * dxinv; 
//    fluxx(i,j,k) = (phi(i+2,j,k)-phi(i+1,j,k)+phi(i,j,k)-phi(i-1,j,k)+phi(i-2,j,k)) * dxinv; 
//    fluxx(i,j,k) = (phi(i+3,j,k)-phi(i+2,j,k)+phi(i+1,j,k)-phi(i,j,k)+phi(i-1,j,k)-phi(i-2,j,k)+phi(i-3,j,k)) * dxinv; 
//    fluxx(i,j,k) = (phi(i+4,j,k)-phi(i+3,j,k)+phi(i+2,j,k)-phi(i+1,j,k)+
//                    phi(i,j,k)-phi(i-1,j,k)+phi(i-2,j,k)-phi(i-3,j,k)+phi(i-4,j,k)) * dxinv; 
//    fluxx(i,j,k) = (phi(i+5,j,k)-phi(i+4,j,k)+phi(i+3,j,k)-phi(i+2,j,k)+phi(i+1,j,k)-
//                    phi(i,j,k)+phi(i-1,j,k)-phi(i-2,j,k)+phi(i-3,j,k)-phi(i-4,j,k)+phi(i-5,j,k)) * dxinv; 
    fluxx(i,j,k) = (phi(i+6,j,k)-phi(i+5,j,k)+phi(i+4,j,k)-phi(i+3,j,k)+phi(i+2,j,k)-phi(i+1,j,k)+
                    phi(i,j,k)-phi(i-1,j,k)+phi(i-2,j,k)-phi(i-3,j,k)+phi(i-4,j,k)-phi(i-5,j,k)+phi(i-6,j,k)) * dxinv; 
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x_shared (int i, int j, int k, const Box& xbx,
                            amrex::Array4<amrex::Real> const& fluxx,
                            amrex::Array4<amrex::Real const> const& phi, amrex::Real dxinv)
{
/*  
    // TWO POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+1];
    int btid = threadIdx.x + 1;  // Increment by one for convienence
                                 // May need an additional cell at 0.

    // Fill shared.
    sm[btid] = phi(i,j,k);
    // If first cell is not a ghost cell, it will need the previous cell.
    if ((btid == 1) && (xbx.contains({i,j,k})))
    {
        sm[0] = phi(i-1,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid] - sm[btid-1]) * dxinv;
    }

    // THREE POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+2];
    int btid = threadIdx.x + 1;  // Increment by one for convienence
                                 // May need an additional cell at 0.

    // Fill shared.
    sm[btid] = phi(i,j,k);

    if ( (threadIdx.x == 0) && (xbx.contains({i,j,k})) )
    {
        sm[0] = phi(i-1,j,k);
    }
    if ((threadIdx.x == AMREX_GPU_MAX_THREADS-1) && (xbx.contains({i,j,k})))
    {
        sm[AMREX_GPU_MAX_THREADS+1] = phi(i+1,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid+1] - 2*sm[btid] + sm[btid-1]) * dxinv;
    }

    // FIVE POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+4];
    int btid = threadIdx.x + 2;

    // Fill shared.
    sm[btid] = phi(i,j,k);

    if ( (threadIdx.x < 2) && (xbx.contains({i,j,k}) || xbx.contains({i-1,j,k})) )
    {
        sm[btid-2] = phi(i-2,j,k);
    }
    if ((threadIdx.x > AMREX_GPU_MAX_THREADS-3) && (xbx.contains({i,j,k}) || xbx.contains({i+1,j,k})) )
    {
        sm[btid+2] = phi(i+2,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid+2] - sm[btid+1] + sm[btid] - sm[btid-1] + sm[btid-2]) * dxinv;
    }

    // SEVEN POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+6];
    int btid = threadIdx.x + 3;

    // Fill shared.
    sm[btid] = phi(i,j,k);

    if ( (threadIdx.x < 3) && ( xbx.contains({i,j,k}) 
                             || xbx.contains({i-1,j,k})
                             || xbx.contains({i-2,j,k})) )
    {
        sm[btid-3] = phi(i-3,j,k);
    }
    if ( (threadIdx.x > AMREX_GPU_MAX_THREADS-4) && ( xbx.contains({i,j,k})
                                                   || xbx.contains({i+1,j,k})
                                                   || xbx.contains({i+2,j,k})) )
    {
        sm[btid+3] = phi(i+3,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid+3] - sm[btid+2] + sm[btid+1] - sm[btid] + 
                        sm[btid-1] - sm[btid-2] + sm[btid-3]) * dxinv;
    }

    // NINE POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+8];
    int btid = threadIdx.x + 4;

    // Fill shared.
    sm[btid] = phi(i,j,k);

    if ( (threadIdx.x < 4) && ( xbx.contains({i,j,k}) 
                             || xbx.contains({i-1,j,k})
                             || xbx.contains({i-2,j,k})
                             || xbx.contains({i-3,j,k})) )
    {
        sm[btid-4] = phi(i-4,j,k);
    }
    if ( (threadIdx.x > AMREX_GPU_MAX_THREADS-5) && ( xbx.contains({i,j,k})
                                                   || xbx.contains({i+1,j,k})
                                                   || xbx.contains({i+2,j,k})
                                                   || xbx.contains({i+3,j,k})) )
    {
        sm[btid+4] = phi(i+4,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid+4] - sm[btid+3] + sm[btid+2] - sm[btid+1] + sm[btid] - 
                        sm[btid-1] + sm[btid-2] - sm[btid-3] + sm[btid-4) * dxinv;
    }

    // ELEVEN POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+10];
    int btid = threadIdx.x + 5;

    // Fill shared.
    sm[btid] = phi(i,j,k);

    if ( (threadIdx.x < 5) && ( xbx.contains({i,j,k}) 
                             || xbx.contains({i-1,j,k})
                             || xbx.contains({i-2,j,k})
                             || xbx.contains({i-3,j,k})
                             || xbx.contains({i-4,j,k})) )

    {
        sm[btid-5] = phi(i-5,j,k);
    }
    if ( (threadIdx.x > AMREX_GPU_MAX_THREADS-6) && ( xbx.contains({i,j,k})
                                                   || xbx.contains({i+1,j,k})
                                                   || xbx.contains({i+2,j,k})
                                                   || xbx.contains({i+3,j,k})
                                                   || xbx.contains({i+4,j,k})) )
    {
        sm[btid+5] = phi(i+5,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid+5] - sm[btid+4] + sm[btid+3] - sm[btid+2] + sm[btid+1] - sm[btid] +
                        sm[btid-1] - sm[btid-2] + sm[btid-3] - sm[btid-4] + sm[btid-5]) * dxinv;
    }
*/

    // THIRTEEN POINT STENCIL
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+12];
    int btid = threadIdx.x + 6;

    // Fill shared.
    sm[btid] = phi(i,j,k);

    if ( (threadIdx.x < 6) && ( xbx.contains({i,j,k}) 
                             || xbx.contains({i-1,j,k})
                             || xbx.contains({i-2,j,k})
                             || xbx.contains({i-3,j,k})
                             || xbx.contains({i-4,j,k})
                             || xbx.contains({i-5,j,k})) )

    {
        sm[btid-6] = phi(i-6,j,k);
    }
    if ( (threadIdx.x > AMREX_GPU_MAX_THREADS-7) && ( xbx.contains({i,j,k})
                                                   || xbx.contains({i+1,j,k})
                                                   || xbx.contains({i+2,j,k})
                                                   || xbx.contains({i+3,j,k})
                                                   || xbx.contains({i+4,j,k})
                                                   || xbx.contains({i+5,j,k})) )
    {
        sm[btid+6] = phi(i+6,j,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (xbx.contains({i,j,k}))
    {
        fluxx(i,j,k) = (sm[btid+6] - sm[btid+5] + sm[btid+4] - sm[btid+3] + sm[btid+2] - sm[btid+1] + sm[btid] - 
                        sm[btid-1] + sm[btid-2] - sm[btid-3] + sm[btid-4] - sm[btid-5] + sm[btid-6]) * dxinv;
    }


}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dyinv)
{
    fluxy(i,j,k) = (phi(i,j,k)-phi(i,j-1,k)) * dyinv;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y_shared (int i, int j, int k, const Box& ybx,
                            amrex::Array4<amrex::Real> const& fluxy,
                            amrex::Array4<amrex::Real const> const& phi, amrex::Real dyinv)
{
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+1];
    int btid = threadIdx.x + 1;  // Increment by one for convienence
                                 // May need an additional cell at 0.

    // Fill shared.
    sm[btid] = phi(i,j,k);
    // If first cell is not a ghost cell, it will need the previous cell.
    if ((btid == 1) && (ybx.contains({i,j,k})))
    {
        sm[0] = phi(i,j-1,k);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (ybx.contains({i,j,k}))
    {
        fluxy(i,j,k) = (sm[btid] - sm[btid-1]) * dyinv;
    }
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dzinv)
{
    fluxz(i,j,k) = (phi(i,j,k)-phi(i,j,k-1)) * dzinv;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z_shared (int i, int j, int k, const Box& zbx,
                            amrex::Array4<amrex::Real> const& fluxz,
                            amrex::Array4<amrex::Real const> const& phi, amrex::Real dzinv)
{
    __shared__ Real sm[AMREX_GPU_MAX_THREADS+1];
    int btid = threadIdx.x + 1;  // Increment by one for convienence
                                 // May need an additional cell at 0.

    // Fill shared.
    sm[btid] = phi(i,j,k);
    // If first cell is not a ghost cell, it will need the previous cell.
    if ((btid == 1) && (zbx.contains({i,j,k})))
    {
        sm[0] = phi(i,j,k-1);
    }

    __syncthreads();
 
    // Only calculate for interior cells 
    if (zbx.contains({i,j,k}))
    {
        fluxz(i,j,k) = (sm[btid] - sm[btid-1]) * dzinv;
    }
}

#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv))
{
    phinew(i,j,k) = phiold(i,j,k)
        + dt * dxinv * (fluxx(i+1,j  ,k  ) - fluxx(i,j,k))
        + dt * dyinv * (fluxy(i  ,j+1,k  ) - fluxy(i,j,k))
#if (AMREX_SPACEDIM > 2)
        + dt * dzinv * (fluxz(i  ,j  ,k+1) - fluxz(i,j,k));
#else
        ;
#endif
}

#endif
