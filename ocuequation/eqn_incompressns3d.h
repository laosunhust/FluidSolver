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

#ifndef __OCU_EQUATION_EQN_INCOMPRESS_NS_3D_H__
#define __OCU_EQUATION_EQN_INCOMPRESS_NS_3D_H__


#include "ocuutil/timestep.h"
#include "ocuutil/direction.h"
#include "ocuequation/equation.h"
#include "ocuequation/sol_project3d.h"
#include "ocuequation/sol_selfadvection3d.h"
#include "ocuequation/sol_passiveadvection3d.h"
#include "ocuequation/parameters.h"
#include "ocuequation/sol_laplaciancent3d.h"
#include "ocustorage/coarraympi.h"

namespace ocu {


template<typename T>
class Eqn_IncompressibleNS3DParams : public Parameters 
{
public:

  int nx, ny, nz;
  double hx, hy, hz;

  double max_divergence;

  BoundaryConditionSet temp_bc;
  BoundaryConditionSet flow_bc;

  DirectionType vertical_direction;

  T gravity;
  T bouyancy;

  Grid3DHost<T>  init_u, init_v, init_w;
  Grid3DHost<T>  init_temp;

  T thermal_diffusion;
  T viscosity;

  InterpolationType advection_scheme;
  TimeStepType time_step;

  double cfl_factor;

  Eqn_IncompressibleNS3DParams() {
    advection_scheme = IT_FIRST_ORDER_UPWIND;
    viscosity = 0;
    thermal_diffusion = 0;
    viscosity = 0;
    max_divergence = 1e-5;
    nx = ny = nz = 0;
    hx = hy = hz = 0;
    cfl_factor = 1;
    gravity = -9.8;
    bouyancy = 1;
    time_step = TS_FORWARD_EULER;
    vertical_direction = DIR_YPOS;
  }

  // automatically pad everything properly
  void init_grids(int nx_val, int ny_val, int nz_val, bool pinned=true) {
    nx = nx_val;
    ny = ny_val;
    nz = nz_val;
    init_u.init(nx+1,ny,nz,1,1,1,pinned,0,1,1);
    init_v.init(nx,ny+1,nz,1,1,1,pinned,1,0,1);
    init_w.init(nx,ny,nz+1,1,1,1,pinned,1,1,0);
    init_temp.init(nx,ny,nz,1,1,1,pinned,1,1,1);
  }
};



template<typename T>
class Eqn_IncompressibleNS3DBase : public Equation {
protected:

  BoundaryConditionSet _thermalbc;

  double _lastdt;
  DirectionType _vertical_direction;

  int _nx, _ny, _nz;
  double _hx, _hy, _hz;

  double _max_divergence;
  double _cfl_factor;

  TimeStepType _time_step;

  T _gravity, _bouyancy; 

public:
  
  //**** PUBLIC STATE ****
  int num_steps;

  //**** MANAGERS ****
  Eqn_IncompressibleNS3DBase();

  //**** PUBLIC INTERFACE ****
  bool set_base_parameters(const Eqn_IncompressibleNS3DParams<T> &params);

  int nx() const { return _nx; }
  int ny() const { return _ny; }
  int nz() const { return _nz; }
  double hx() const { return _hx; }
  double hy() const { return _hy; }
  double hz() const { return _hz; }
};




template<typename T>
class Eqn_IncompressibleNS3D : public Eqn_IncompressibleNS3DBase<T> {
protected:
  //**** MEMBER VARIABLES ****
  Sol_ProjectDivergence3DDevice<T> _projection_solver; // contains pressure
  Sol_SelfAdvection3DDevice<T>     _advection_solver; // contains dudt, dvdt, dwdt
  Sol_PassiveAdvection3DDevice<T>  _thermal_solver; // contains temperature state
  Sol_LaplacianCentered3DDevice<T> _thermal_diffusion; // calculates thermal diffusion
  Sol_LaplacianCentered3DDevice<T> _u_diffusion; // calculates u part of viscosity
  Sol_LaplacianCentered3DDevice<T> _v_diffusion; // calculates v part of viscosity
  Sol_LaplacianCentered3DDevice<T> _w_diffusion; // calculates w part of viscosity

  Grid3DDevice<T>  _u, _v, _w;
  Grid3DDevice<T>  _deriv_udt, _deriv_vdt, _deriv_wdt;

  Grid3DDevice<T>  _temp;
  Grid3DDevice<T>  _deriv_tempdt;

  // for AB2 stepper
  Grid3DDevice<T>  _last_deriv_tempdt;
  Grid3DDevice<T>  _last_deriv_udt, _last_deriv_vdt, _last_deriv_wdt;

  void add_thermal_force();

public:
  

  //**** MANAGERS ****
  Eqn_IncompressibleNS3D();

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_IncompressibleNS3DParams<T> &params);

  double get_max_stable_timestep() const;
  bool advance_one_step(double dt);

  T viscosity_coefficient()         const { return _u_diffusion.coefficient; }
  T thermal_diffusion_coefficient() const { return _thermal_diffusion.coefficient; }

  const Grid3DDevice<T> &get_u() const { return _u; }
  const Grid3DDevice<T> &get_v() const { return _v; }
  const Grid3DDevice<T> &get_w() const { return _w; }
  const Grid3DDevice<T> &get_temperature() const { return _temp; }
};



template<typename T>
class Eqn_IncompressibleNS3DCo : public Eqn_IncompressibleNS3DBase<T> {
protected:

  int _u_negx_hdl;
  int _v_negx_hdl;
  int _w_negx_hdl;
  int _t_negx_hdl;

  int _u_posx_hdl;
  int _v_posx_hdl;
  int _w_posx_hdl;
  int _t_posx_hdl;

  //**** MEMBER VARIABLES ****
  Sol_ProjectDivergence3DDeviceCo<T> _projection_solver; // contains pressure
  Sol_SelfAdvection3DDevice<T>     _advection_solver; // contains dudt, dvdt, dwdt
  Sol_PassiveAdvection3DDevice<T>  _thermal_solver; // contains temperature state
  Sol_LaplacianCentered3DDevice<T> _thermal_diffusion; // calculates thermal diffusion
  Sol_LaplacianCentered3DDevice<T> _u_diffusion; // calculates u part of viscosity
  Sol_LaplacianCentered3DDevice<T> _v_diffusion; // calculates v part of viscosity
  Sol_LaplacianCentered3DDevice<T> _w_diffusion; // calculates w part of viscosity

  Grid3DDeviceCo<T>  _u, _v, _w;
  Grid3DDeviceCo<T>  _deriv_udt, _deriv_vdt, _deriv_wdt;

  Grid3DDeviceCo<T>  _temp;
  Grid3DDeviceCo<T>  _deriv_tempdt;

  // for AB2 stepper
  Grid3DDevice<T>  _last_deriv_tempdt;
  Grid3DDevice<T>  _last_deriv_udt, _last_deriv_vdt, _last_deriv_wdt;


  BoundaryConditionSet _local_thermalbc;
  BoundaryConditionSet _local_bc;

  void add_thermal_force();
  void do_halo_exchange_uvw();
  void do_halo_exchange_t();

public:
  

  //**** MANAGERS ****
  Eqn_IncompressibleNS3DCo(const char *name);
  ~Eqn_IncompressibleNS3DCo();

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_IncompressibleNS3DParams<T> &params);

  double get_max_stable_timestep() const;
  bool advance_one_step(double dt);

  T viscosity_coefficient()         const { return _u_diffusion.coefficient; }
  T thermal_diffusion_coefficient() const { return _thermal_diffusion.coefficient; }

  const Grid3DDeviceCo<T> &get_u() const { return _u; }
  const Grid3DDeviceCo<T> &get_v() const { return _v; }
  const Grid3DDeviceCo<T> &get_w() const { return _w; }
  const Grid3DDeviceCo<T> &get_temperature() const { return _temp; }
};
/*
template<typename T>
class Eqn_IncompressibleNS3DMPI : public Eqn_IncompressibleNS3DBase<T> {
protected:

  // **** MEMBER VARIABLES ****
  
  Sol_ProjectDivergence3DDeviceCo<T> _projection_solver; //  Pressure solver for MPI version underdevelopment
  
  // **** TO DO *****
  // Presure solver for MPI version


  Sol_SelfAdvection3DDevice<T>     _advection_solver; // contains dudt, dvdt, dwdt
  //Sol_PassiveAdvection3DDevice<T>  _thermal_solver; // contains temperature state
  //Sol_LaplacianCentered3DDevice<T> _thermal_diffusion; // calculates thermal diffusion
  Sol_LaplacianCentered3DDevice<T> _u_diffusion; // calculates u part of viscosity
  Sol_LaplacianCentered3DDevice<T> _v_diffusion; // calculates v part of viscosity
  Sol_LaplacianCentered3DDevice<T> _w_diffusion; // calculates w part of viscosity

  Grid3DDevice<T>  _u, _v, _w;
  Grid3DDevice<T>  _deriv_udt, _deriv_vdt, _deriv_wdt;

  //Grid3DDevice<T>  _temp;
  //Grid3DDevice<T>  _deriv_tempdt;

  // for AB2 stepper
  //Grid3DDevice<T>  _last_deriv_tempdt;
  Grid3DDevice<T>  _last_deriv_udt, _last_deriv_vdt, _last_deriv_wdt;


  //BoundaryConditionSet _local_thermalbc;
  BoundaryConditionSet _local_bc;

  //void add_thermal_force(); 
  CoarrayMPIManager    mpimanager_main; 
  CoarrayMPIManager    mpimanager_u; 
  CoarrayMPIManager    mpimanager_v; 
  CoarrayMPIManager    mpimanager_w; 
  void do_halo_exchange_uvw_mpi();
  //void do_halo_exchange_t();

public:
  

  //  **** MANAGERS ****
  Eqn_IncompressibleNS3DMPI(CoarrayMPIManager &mpiinfo);
  ~Eqn_IncompressibleNS3DMPI();
  
  // **** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_IncompressibleNS3DParams<T> &params);

  double get_max_stable_timestep() const;
  bool advance_one_step(double dt);

  T viscosity_coefficient()         const { return _u_diffusion.coefficient; }
  //T thermal_diffusion_coefficient() const { return _thermal_diffusion.coefficient; }

  const Grid3DDevice<T> &get_u() const { return _u; }
  const Grid3DDevice<T> &get_v() const { return _v; }
  const Grid3DDevice<T> &get_w() const { return _w; }
  //const Grid3DDevice<T> &get_temperature() const { return _temp; }
};
*/
typedef Eqn_IncompressibleNS3D<float> Eqn_IncompressibleNS3DF;
typedef Eqn_IncompressibleNS3D<double> Eqn_IncompressibleNS3DD;
typedef Eqn_IncompressibleNS3DCo<float> Eqn_IncompressibleNS3DCoF;
typedef Eqn_IncompressibleNS3DCo<double> Eqn_IncompressibleNS3DCoD;
typedef Eqn_IncompressibleNS3DParams<float> Eqn_IncompressibleNS3DParamsF;
typedef Eqn_IncompressibleNS3DParams<double> Eqn_IncompressibleNS3DParamsD;


//typedef Eqn_IncompressibleNS3DMPI<float> Eqn_IncompressibleNS3DMPIF;
//typedef Eqn_IncompressibleNS3DMPI<double> Eqn_IncompressibleNS3DMPID;

} // end namespace

#endif

