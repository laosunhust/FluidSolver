#include "tests/testframework.h"
#include "ocuutil/boundary_condition.h"
#include "ocuequation/eqn_diffusion3d.h"
#include "ocuequation/eqn_advectiondiffusion3d.h"
#include "ocuutil/float_routines.h"


using namespace ocu;

DECLARE_UNITTEST_BEGIN(AdvectionDiffusion3DTest);
void run_test(int nx, int ny, int nz, float hx, float hy, float hz, BoundaryConditionSet bc, bool no_border_contrib)
{
  Eqn_AdvectionDiffusion3DParams<float> params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.hx = hx;
  params.hy = hy;
  params.hz = hz;
  params.flow_bc = bc;  
  params.viscosity = 5;
  //params.init_u.init(nx, ny, nz, 1, 1, 1);
  //params.init_v.init(nx, ny, nz, 1, 1, 1);
  //params.init_w.init(nx, ny, nz, 1, 1, 1);
  params.init_grids(nx,ny,nz);
  float midpt_x = (nx-1) * hx * .5;
  float midpt_y = (ny-1) * hy * .5;
  float midpt_z = (nz-1) * hz * .5;
  float mindim = min3(nx*hx, ny*hy, nz*hz);

  int i,j,k;
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        float px = i * hx;
        float py = j * hy;
        float pz = k * hz;

        float rad = sqrt((px - midpt_x)*(px - midpt_x) + (py - midpt_y)*(py - midpt_y) + (pz - midpt_z)*(pz - midpt_z));
        if (rad < (mindim/4)) {
          params.init_u.at(i,j,k) = 1;
        }
        else {
          params.init_u.at(i,j,k) = 0;
        }
        params.init_v.at(i,j,k) = 1;
        params.init_w.at(i,j,k) = 1;
      }
  //float variation_before;
  //float integral_before;

  //params.initial_values.reduce_sqrsum(variation_before);
  //params.initial_values.reduce_sum(integral_before);

  Eqn_AdvectionDiffusion3D<float> eqn;
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));
  
  // diffuse it.
  UNITTEST_ASSERT_TRUE(eqn.advance_one_step(.1));
  float check_nan;
  eqn.get_u().reduce_checknan(check_nan);
  UNITTEST_ASSERT_FINITE(check_nan);
  eqn.get_v().reduce_checknan(check_nan);
  UNITTEST_ASSERT_FINITE(check_nan);
  eqn.get_w().reduce_checknan(check_nan);
  UNITTEST_ASSERT_FINITE(check_nan);
  /*  
  if (no_border_contrib) {
    // verify TVD
    // verify conservative
    float variation_after;
    float integral_after;
    eqn.density().reduce_sqrsum(variation_after);
    eqn.density().reduce_sum(integral_after);
    UNITTEST_ASSERT_EQUAL_DOUBLE(integral_after, integral_before, .01f);
    UNITTEST_ASSERT_TRUE(variation_after <= variation_before);
  }
  */ 

}

void run_all_bcs(int nx, int ny, int nz, float hx, float hy, float hz) {
  BoundaryCondition example;
  BoundaryConditionSet bc;

  example.type = BC_PERIODIC;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, true);

  example.type = BC_NEUMANN;
  example.value = 0;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, true);

  example.type = BC_NEUMANN;
  example.value = 1;
  bc = BoundaryCondition(example);
  // change signs for positive sides so it will be symmetric
  bc.xpos.value = -1;
  bc.ypos.value = -1;
  bc.zpos.value = -1;
  run_test(nx, ny, nz, hx, hy, hz, bc, false);

  example.type = BC_DIRICHELET;
  example.value = 0;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, true);

  example.type = BC_DIRICHELET;
  example.value = 1;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, false);
}


void run()
{
  run_all_bcs(128, 128, 128, 1, 1, 1);
  //run_all_bcs(128, 128, 128, .5, .3, 1);
  //run_all_bcs(60, 128, 128, .5, .3, 1);
  //run_all_bcs(128, 60, 128, .5, .7, 1);
  //run_all_bcs(128, 128, 60, .5, .3, 1);
  //run_all_bcs(34, 57, 92, 1, 1, 1);
  //run_all_bcs(39, 92, 57, 1, 1, 1);
 
}
DECLARE_UNITTEST_END(AdvectionDiffusion3DTest);

