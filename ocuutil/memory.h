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

#ifndef __OCU_UTIL_MEMORY_H__
#define __OCU_UTIL_MEMORY_H__


namespace ocu {

enum MemoryType {
  MEM_INVALID,
  MEM_HOST,
  MEM_DEVICE,
  // zero-copied, mpi interop, write-combined, portable pinned, remote host, remote device, etc.
};

void *host_malloc(size_t bytes, bool pinned, bool write_combined = false);
void host_free(void *, bool pinned_or_write_combined);

} // end namespace


#endif

