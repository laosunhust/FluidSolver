/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/grid1d.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/gridnetcdf.h"
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"
#include "ocustorage/coarray.h"
#include "ocuutil/thread.h"

#include <algorithm>

using namespace ocu;

#ifdef OCU_DOUBLESUPPORT
typedef double T;
#else
typedef float T;
#endif



//const double convergence_tol = 1e-6;
const double convergence_tol = 1e-4;

#include "ocuutil/color.h"
#include "ocuutil/imagefile.h"

void write_slice(const char *filename, const Grid3DHostCo<T> &grid, int frame)
{
  char buff[1024];
  sprintf(buff, "%s.%04d.ppm", filename, frame);

  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();

  ImageFile img;
  img.allocate(nx*ThreadManager::num_images(), ny);

  for (int co=0; co < ThreadManager::num_images(); co++)
    for (int i=0; i < nx; i++)
      for (int j=0; j < ny; j++) {
        const Grid3DHostCo<T> *nbr = grid.co(co);
        T temperature = nbr->at(i,j,nz/2);
        if (temperature < -2) temperature = -2;
        if (temperature > 2)  temperature = 2;
        //float3 color = make_float3(temperature, temperature, temperature);
        float3 color = hsv_to_rgb(make_float3((temperature + 2)*90, 1, 1));
        //float3 color = pseudo_temperature((temperature+1)*.5);
        img.set_rgb(i+co*nx,j,(unsigned char)(255*color.x),(unsigned char)(255*color.y),(unsigned char)(255*color.z));
      }

  img.write_ppm(buff);
}


void write_slice(const char *filename, const Grid3DDevice<T> &grid)
{
  Grid3DHost<T> h_grid;
  h_grid.init_congruent(grid);
  h_grid.copy_all_data(grid);


  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();

  ImageFile img;
  img.allocate(nx, ny);

  for (int i=0; i < nx; i++)
    for (int j=0; j < ny; j++) {
      T temperature = h_grid.at(i,j,nz/2);
      if (temperature < -2) temperature = -2;
      if (temperature > 2)  temperature = 2;
      //float3 color = make_float3(temperature, temperature, temperature);
      float3 color = hsv_to_rgb(make_float3((temperature + 2)*90, 1, 1));
      //float3 color = pseudo_temperature((temperature+1)*.5);
      img.set_rgb(i,j,(unsigned char)(255*color.x),(unsigned char)(255*color.y),(unsigned char)(255*color.z));
    }

  img.write_ppm(filename);
}

void init_params(Eqn_IncompressibleNS3DParams<T> &params, int nx, int ny, int nz, int image_id, int num_images)
{
  params.init_grids(nx, ny, nz);
  params.hx = 100.0f/(nx*num_images);
  params.hy = 100.0f/ny;
  params.hz = 100.0f/nz;
  params.bouyancy = .2;
  params.gravity = -9.8;

  params.max_divergence = convergence_tol;
  
  // these all go together:
  params.viscosity = 1;
  params.thermal_diffusion = 1;
  params.advection_scheme = IT_SECOND_ORDER_CENTERED;
  params.time_step = TS_ADAMS_BASHFORD2;
  params.cfl_factor = .7;

  /*
  params.viscosity = 0;
  params.thermal_diffusion = 0;
  params.advection_scheme = IT_FIRST_ORDER_UPWIND;
  params.time_step = TS_FORWARD_EULER;
  params.cfl_factor = .99;
  */

  BoundaryCondition closed;
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  params.flow_bc = BoundaryConditionSet(closed);

  BoundaryCondition neumann;
  neumann.type = BC_NEUMANN;
  params.temp_bc = BoundaryConditionSet(neumann);

  int i,j,k;
  params.init_u.clear_zero();
  params.init_v.clear_zero();
  params.init_w.clear_zero();
  
  float midpt_x = nx * params.hx * num_images * .5;
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        float x_pos = (i+.5+(image_id * nx))*params.hx;        
        params.init_temp.at(i,j,k) = -1 + 2 * tanh(.1*(x_pos - midpt_x));
      }
}

void run_sim(int nx, int ny, int nz, const char *out_file, float endtime)
{
  Eqn_IncompressibleNS3DParams<T> params;
  Eqn_IncompressibleNS3D<T> eqn;

  init_params(params, nx, ny, nz,0,1);

  if (!eqn.set_parameters(params)) {
    printf("[ERROR] Could not initialize simulation\n");
    exit(-1);
  }


  double dt = .1;
  double this_dt = .1;

  CPUTimer timer;
  double total_time = 0;

  int steps = 0;
  timer.start();
  int next_frame = 1;

  NetCDFGrid3DWriter file_writer;
  if (out_file) {
    file_writer.open(out_file, eqn.nx(), eqn.ny(), eqn.nz(), eqn.hx(), eqn.hy(), eqn.hz());
    file_writer.define_variable("temperature", NC_DOUBLE, GS_CENTER_POINT);
    file_writer.define_variable("u", NC_DOUBLE, GS_U_FACE);
    file_writer.define_variable("v", NC_DOUBLE, GS_V_FACE);
    file_writer.define_variable("w", NC_DOUBLE, GS_W_FACE);
  }

  Grid3DHost<T> h_temp, h_u, h_v, h_w;
  h_temp.init_congruent(eqn.get_temperature());
  h_u.init_congruent(eqn.get_u());
  h_v.init_congruent(eqn.get_v());
  h_w.init_congruent(eqn.get_w());

  size_t time_level = 0;

  for (float t=0; t < endtime; t += this_dt) {
    printf("t = %f (step %d) \n**************\n", t, steps);

    timer.start();
    this_dt = std::min(eqn.get_max_stable_timestep(), dt);
    eqn.advance_one_step(this_dt);
    timer.stop();

    total_time += timer.elapsed_sec();
    printf(".");
    fflush(stdout);

    // write the output
    if (out_file && t > time_level * dt) {
      file_writer.add_time_level(t, time_level);

      h_temp.copy_all_data(eqn.get_temperature());
      h_u.copy_all_data(eqn.get_u());
      h_v.copy_all_data(eqn.get_v());
      h_w.copy_all_data(eqn.get_w());
      file_writer.add_data("temperature", h_temp, time_level);
      file_writer.add_data("u", h_u, time_level);
      file_writer.add_data("v", h_v, time_level);
      file_writer.add_data("w", h_w, time_level);
    }

    steps++;
  }

  if (out_file) {
    file_writer.close();
  }

  printf("\nTotal elapsed: %f sec, %d steps, %f sec/step\n", total_time, steps, total_time/steps);
  global_timer_print();
}


void run_sim_multi(int nx, int ny, int nz, const char *out_file, float endtime)
{
  int num_gpus=2;

  CoArrayManager::initialize(num_gpus);
  ThreadManager::initialize(num_gpus);

#pragma omp parallel 
  {
    CoArrayManager::initialize_image();
    ThreadManager::initialize_image();

    Eqn_IncompressibleNS3DParams<T> params;
    Eqn_IncompressibleNS3DCo<T> eqn("eqn");

    init_params(params, nx, ny, nz,ThreadManager::this_image(),num_gpus);


    if (!eqn.set_parameters(params)) {
      printf("[ERROR] Could not initialize simulation\n");
      exit(-1);
    }



    double dt = .1;
    double this_dt = .1;

    CPUTimer timer;
    double total_time = 0;

    int steps = 0;
    timer.start();
    int next_frame = 1;

    Grid3DHostCo<T> h_temp("h_temp");
    h_temp.init(nx,ny,1,0,0,0);

    
    int hdl_d_to_h = CoArrayManager::barrier_allocate(h_temp.region()()(), eqn.get_temperature().region(0,nx-1)(0,ny-1)(nz/2));

    for (float t=0; t < endtime; t += this_dt) {

      printf("t = %f (step %d) \n**************\n", t, steps);
      timer.start();
      this_dt = std::min(eqn.get_max_stable_timestep(), dt);
      eqn.advance_one_step(this_dt);
      timer.stop();

      total_time += timer.elapsed_sec();

      if (out_file && t > next_frame * dt) {
        CoArrayManager::barrier_exchange(hdl_d_to_h);        
        CoArrayManager::barrier_exchange_fence();        
        ThreadManager::barrier();
        next_frame++;
      }

      if (ThreadManager::this_image() == 0) {
        printf(".");
        fflush(stdout);

        if (out_file && t > (next_frame-1) * dt) {
          // write the output
          write_slice(out_file, h_temp, next_frame-1); 
       }
      }

      steps++;
    }

    if (ThreadManager::this_image() == 0) {
      printf("\nTotal elapsed: %f sec, %d steps, %f sec/step\n", total_time, steps, total_time/steps);
      global_timer_print();
    }

    ThreadManager::barrier();
  }
}

int main(int argc, const char **argv)
{
  if (argc < 4) {
    printf("[usage] lockex [-multi] [-time End] nx ny nz [outfile]");
    exit(-1);
  }

  int next_arg = 1;
  bool do_multi = false;
  if (strcmp(argv[next_arg], "-multi") == 0) {
    do_multi = true;
    next_arg++;
  }

  float endtime = 10.0f;
  if (strcmp(argv[next_arg], "-time") == 0) {
    endtime = atof(argv[next_arg+1]);
    next_arg += 2;
  }

  int nx = atoi(argv[next_arg++]);
  int ny = atoi(argv[next_arg++]);
  int nz = atoi(argv[next_arg++]);

  const char *out_file = 0;
  if (argc >= next_arg-1)
    out_file = argv[next_arg++];

  if (do_multi)
    run_sim_multi(nx, ny, nz, out_file, endtime);
  else
    run_sim(nx, ny, nz, out_file, endtime);

  return 0;
}
