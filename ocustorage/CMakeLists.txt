#  Copyright 2008-2009 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License



CUDA_ADD_LIBRARY(ocustorage 
  grid1ddev.cu
  grid1dhost.cu
  grid1d.h
  grid1dops.h
  grid1dops.cu
  grid2dhost.cpp
  grid2ddev.cu
  grid2dhost.cu
  grid2d.h
  grid3ddev.cu
  grid3dhost.cu
  grid3d.h
  grid3d.cpp
  grid3dops.h
  grid3dops.cpp
  grid3dops.cu
  grid3dsample.h
  grid3dsample.cu
  grid3dboundary.h
  grid3dboundary.cu
  gridnetcdf.h
  gridnetcdf.cpp
  region1d.h
  region3d.h
  coarray.h
  coarray_internal.h
  coarray.cu
  coarraympi.h
  coarraympi.cpp
  barrier_exchange.cu
  exchange_mempool.cu
  transfer_q.cu
  transfer_routines.cu
  #mpistructs.h
  #mpistructs.cpp
) 

TARGET_LINK_LIBRARIES(ocustorage ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(ocustorage PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(ocustorage PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

