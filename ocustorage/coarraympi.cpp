#include "coarraympi.h"
#include <mpi.h> 
#include <assert.h>
using namespace std;
using namespace ocu;

ocu::CoarrayMPIConfig::CoarrayMPIConfig(int xn, int yn, int zn) {
	npe.resize(3);
	npe[0] = xn;
	npe[1] = yn;
	npe[2] = zn;
}
void ocu::CoarrayMPIConfig::set(int xn, int yn, int zn) {
	npe.resize(3);
	npe[0] = xn;
	npe[1] = yn;
	npe[2] = zn;
}
ocu::CoarrayMPIConfig::CoarrayMPIConfig() {
	npe.resize(3);
	npe[0] = -1;
	npe[1] = -1;
	npe[2] = -1;
}

ocu::CoarrayMPIManager::CoarrayMPIManager(int myrank,
		CoarrayMPIConfig &mpiconfig) {
	gridconfig = mpiconfig;
	mympirank = myrank;
	determineNeighbors();
	//std::cout<<"Manager constructed for node "<<myrank<<std::endl;
}

ocu::CoarrayMPIManager::CoarrayMPIManager(int myrank, int totalNumProc,
		Global_Grid compuGrid, int buf_dep) {

	ggr = compuGrid;
	systemsize = totalNumProc;
	mympirank = myrank;
	buffer_depth = buf_dep;

	generateMpiConfig();
	calculateGridDimension();
	determineNeighbors();
	calculateBufferSize();
	allocateBuffer();

	//std::cout<<"Manager constructed for node "<<myrank<<std::endl;
}

void ocu::CoarrayMPIManager::generateMpiConfig() {
	if (ps.type() == 0) {
		assert(ggr.globalx >= ggr.globaly && ggr.globalx >= ggr.globalz);
		gridconfig.set(systemsize, 1, 1);
	} else if (ps.type() == 1) {
		gridconfig.set(ps.xd(), ps.yd(), ps.zd());
		//cout<<"processor grid dimension: ["<< ps.xd()<<","<<ps.yd()<<","<<ps.zd()<<"]"<<endl;
	} else {

	}
}

ocu::CoarrayMPIManager::CoarrayMPIManager(int myrank, int totalsize,
		Partition_strategy &pst, Global_Grid &glgr, int buf_depth) {
	mympirank = myrank;
	ps = pst;
	ggr = glgr;
	systemsize = totalsize;
	buffer_depth = buf_depth;
	//gridconfig = generate_config();
	generateMpiConfig();
	calculateGridDimension();

	determineNeighbors();
	calculateBufferSize();
	allocateBuffer();
	mygrid.init(xsize, ysize, zsize, buffer_depth, buffer_depth, buffer_depth);
	mygrid.clear(mympirank);
	//std::cout<<"Manager constructed for node "<<myrank<<std::endl;
}

ocu::CoarrayMPIManager::CoarrayMPIManager(int myrank, int totalsize,
		Partition_strategy &pst, Global_Grid &glgr, Grid3DDeviceD *devdata) {
	assert(devdata->gx() == devdata->gy() && devdata->gy() == devdata->gz());
	mympirank = myrank;
	ps = pst;
	ggr = glgr;
	systemsize = totalsize;
	buffer_depth = devdata->gx();
	//gridconfig = generate_config();
	generateMpiConfig();
	calculateGridDimension();

	determineNeighbors();
	calculateBufferSize();
	allocateBuffer();
	//mygrid.init(xsize,ysize,zsize,buffer_depth,buffer_depth,buffer_depth);
	//mygrid.clear(mympirank);
	//std::cout<<"Manager constructed for node "<<myrank<<std::endl;
}

void ocu::CoarrayMPIManager::allocateBuffer() {

	cout<< "buffersizeinfo: " <<endl;
	cout<<"X direction: (" <<xn_buffersize<<","<<xp_buffersize<<")"<<endl;
	cout<<"Y direction: (" <<yn_buffersize<<","<<yp_buffersize<<")"<<endl;
	cout<<"Z direction: (" <<zn_buffersize<<","<<zp_buffersize<<")"<<endl;

	XN_buffer_send = new double[xn_buffersize];
	XP_buffer_send = new double[xp_buffersize];
	YN_buffer_send = new double[yn_buffersize];
	YP_buffer_send = new double[yp_buffersize];
	ZN_buffer_send = new double[zn_buffersize];
	ZP_buffer_send = new double[zp_buffersize];

	XN_buffer_recv = new double[xn_buffersize];
	XP_buffer_recv = new double[xp_buffersize];
	YN_buffer_recv = new double[yn_buffersize];
	YP_buffer_recv = new double[yp_buffersize];
	ZN_buffer_recv = new double[zn_buffersize];
	ZP_buffer_recv = new double[zp_buffersize];
	MPI_Barrier(MPI_COMM_WORLD );
}
ocu::CoarrayMPIManager::~CoarrayMPIManager() {
	delete XN_buffer_send;
	delete XP_buffer_send;
	delete YN_buffer_send;
	delete YP_buffer_send;
	delete ZN_buffer_send;
	delete ZP_buffer_send;
	delete XN_buffer_recv;
	delete XP_buffer_recv;
	delete YN_buffer_recv;
	delete YP_buffer_recv;
	delete ZN_buffer_recv;
	delete ZP_buffer_recv;
}
void ocu::CoarrayMPIManager::calculateGridDimension() {
	// by default, partition x direction only
	if (ps.type() == 0) {
		assert(ggr.globalx >= ggr.globaly && ggr.globalx >= ggr.globalz);
		//vector<int> result;
		//result.resize(3);
		int res_truncated = ggr.globalx / systemsize;
		int numofproc_has1 = ggr.globalx - res_truncated * systemsize;
		if (mympirank < numofproc_has1) {
			xsize = res_truncated + 1;
			xn = xsize * mympirank;
			xp = xn + xsize;
		} else {
			xsize = res_truncated;
			xn = (res_truncated + 1) * numofproc_has1
					+ res_truncated * (mympirank - numofproc_has1);
			xp = xn + xsize;
		}
		ysize = ggr.globaly;
		yn = 0;
		yp = ggr.globaly;
		zsize = ggr.globalz;
		zn = 0;
		zp = ggr.globalz;
	} else if (ps.type() == 1) {
		int plane_size = gridconfig.getnpy() * gridconfig.getnpz();
		int line_size = gridconfig.getnpz();
		int myxpos, myypos, myzpos;
		int tmpx, tmpy, tmpz;
		int tmp;
		myxpos = mympirank / plane_size;
		tmp = mympirank - myxpos * plane_size;
		myypos = tmp / line_size;
		tmp = tmp - myypos * line_size;
		myzpos = tmp;
		int res_truncated_x = ggr.globalx / ps.xd();
		int numofproc_has1x = ggr.globalx - res_truncated_x * ps.xd();
		if (myxpos < numofproc_has1x) {
			xsize = res_truncated_x + 1;
			xn = xsize * myxpos;
			xp = xn + xsize;
		} else {
			xsize = res_truncated_x;
			xn = (res_truncated_x + 1) * numofproc_has1x
					+ res_truncated_x * (myxpos - numofproc_has1x);
			xp = xn + xsize;
		}

		int res_truncated_y = ggr.globaly / ps.yd();
		int numofproc_has1y = ggr.globaly - res_truncated_y * ps.yd();
		if (myypos < numofproc_has1y) {
			ysize = res_truncated_y + 1;
			yn = ysize * myypos;
			yp = yn + ysize;
		} else {
			ysize = res_truncated_y;
			yn = (res_truncated_y + 1) * numofproc_has1y
					+ res_truncated_y * (myypos - numofproc_has1y);
			yp = yn + ysize;
		}

		int res_truncated_z = ggr.globalz / ps.zd();
		int numofproc_has1z = ggr.globalz - res_truncated_z * ps.zd();
		if (myzpos < numofproc_has1z) {
			zsize = res_truncated_z + 1;
			zn = zsize * myzpos;
			zp = zn + zsize;
		} else {
			zsize = res_truncated_z;
			zn = (res_truncated_z + 1) * numofproc_has1z
					+ res_truncated_z * (myypos - numofproc_has1z);
			zp = zn + zsize;
		}

	} else {

	}
}
;

void ocu::CoarrayMPIManager::printMyInfo() {
	cout << "Using partition strategy: " << ps.type() << endl;
	cout << "There are " << systemsize << " processors globally" << endl;
	cout << "Global grid dimension is [" << ggr.globalx << "," << ggr.globaly
			<< "," << ggr.globalz << "]" << endl;
	cout << "I am " << mympirank
			<< "th processor and my computational domain is from X:[" << XN()
			<< "," << XP() << "), Y:[" << YN() << "," << YP() << "),Z:[" << ZN()
			<< "," << ZP() << ")" << endl;
}

bool ocu::CoarrayMPIManager::determineNeighbors() {
	// rank processors in the order of x, y, z
	if (!gridconfig.is_valid()) {
		return 0;
	} else {
		int plane_size = gridconfig.getnpy() * gridconfig.getnpz();
		int line_size = gridconfig.getnpz();
		int myxpos, myypos, myzpos;
		int tmpx, tmpy, tmpz;
		int tmp;
		myxpos = mympirank / plane_size;
		tmp = mympirank - myxpos * plane_size;
		myypos = tmp / line_size;
		tmp = tmp - myypos * line_size;
		myzpos = tmp;
		//cout<<"I am processor "<<mympirank<<" and my position is ["<<myxpos<<","<<myypos<<","<<myzpos<<"]"<<endl;
		tmpx = myxpos - 1;
		tmpy = myypos;
		tmpz = myzpos;
		if (tmpx < 0) {
			tmpx = gridconfig.getnpx() + tmpx;
		}
		neigh_xn = tmpx * plane_size + tmpy * line_size + tmpz;
		tmpx = myxpos + 1;
		tmpy = myypos;
		tmpz = myzpos;
		if (tmpx >= gridconfig.getnpx()) {
			tmpx = tmpx % gridconfig.getnpx();
		}
		neigh_xp = tmpx * plane_size + tmpy * line_size + tmpz;

		tmpx = myxpos;
		tmpy = myypos - 1;
		tmpz = myzpos;
		if (tmpy < 0) {
			tmpy = gridconfig.getnpy() + tmpy;
		}
		neigh_yn = tmpx * plane_size + tmpy * line_size + tmpz;
		tmpx = myxpos;
		tmpy = myypos + 1;
		tmpz = myzpos;
		if (tmpy >= gridconfig.getnpy()) {
			tmpy = tmpy % gridconfig.getnpy();
		}
		neigh_yp = tmpx * plane_size + tmpy * line_size + tmpz;

		tmpx = myxpos;
		tmpy = myypos;
		tmpz = myzpos - 1;
		if (tmpz < 0) {
			tmpz = gridconfig.getnpz() + tmpz;
		}
		neigh_zn = tmpx * plane_size + tmpy * line_size + tmpz;
		tmpx = myxpos;
		tmpy = myypos;
		tmpz = myzpos + 1;
		if (tmpz >= gridconfig.getnpz()) {
			tmpz = tmpz % gridconfig.getnpz();
		}
		neigh_zp = tmpx * plane_size + tmpy * line_size + tmpz;
		cout<<"X neighbors: negative: "<< neigh_xn<<" positive: "<<neigh_xp<<"\n"
		    <<"Y neighbors: negative: "<< neigh_yn<<" positive: "<<neigh_yp<<"\n"
		    <<"Z neighbors: negative: "<< neigh_zn<<" positive: "<<neigh_zp<<endl;
		return 1;
	}
}

//void ocu::printfirstandlastinbuffer(double* buffer, int buffersize)
//{

//}

bool ocu::CoarrayMPIManager::dataExchangeWithNeighbors() {
	//cout<<" for processor "<<mympirank<<", grid before exchange :"<<endl;
	//mygrid.printallelements();
	// Follow data exchange pattern of X negative, X positive , Y negative, Y positive, Z negative, Z positive
	MPI_Barrier(MPI_COMM_WORLD );
	Direction dir = XNdir;
	if (dir != DIR_NONE) {
		datapro.pack_data(dir, XN_buffer_send, xn_buffersize, &mygrid,
				mympirank);
		cout<<" processor rank :"<<mympirank<<" XN buffer send begin: "<<XN_buffer_send[0]<<" end :"<<XN_buffer_send[xn_buffersize-1]<<endl;
		comm.coarray_send_interior_states(mympirank, neigh_xn, dir,
				XN_buffer_send, xn_buffersize);
		comm.coarray_recv_interior_states(neigh_xp, mympirank, dir,
				XP_buffer_recv, xp_buffersize);
		datapro.unpack_data(dir, XP_buffer_recv, xp_buffersize, &mygrid,
				mympirank);
		//cout<<" processor rank :"<<mympirank<<" XP buffer recv begin: "<<XP_buffer_recv[0]<<" end :"<<XP_buffer_recv[xp_buffersize-1]<<endl;
	}

	MPI_Barrier(MPI_COMM_WORLD );
	dir = XPdir;
	if (dir != DIR_NONE) {
		//cout<<" before packing data , proc #:"<<mympirank<<endl;
		datapro.pack_data(dir, XP_buffer_send, xp_buffersize, &mygrid,
				mympirank);
		comm.coarray_send_interior_states(mympirank, neigh_xp, dir,
				XP_buffer_send, xp_buffersize);
		comm.coarray_recv_interior_states(neigh_xn, mympirank, dir,
				XN_buffer_recv, xn_buffersize);
		datapro.unpack_data(dir, XN_buffer_recv, xn_buffersize, &mygrid,
				mympirank);
		//cout<<" for processor "<<mympirank<<", grid before exchange :"<<endl;
		//mygrid.printallelements();
	}

	MPI_Barrier(MPI_COMM_WORLD );
	dir = YNdir;
	if (dir != DIR_NONE) {
		//Direction dir = YNdir;
		datapro.pack_data(dir, YN_buffer_send, yn_buffersize, &mygrid,
				mympirank);
		comm.coarray_send_interior_states(mympirank, neigh_yn, dir,
				YN_buffer_send, yn_buffersize);
		comm.coarray_recv_interior_states(neigh_yp, mympirank, dir,
				YP_buffer_recv, yp_buffersize);
		datapro.unpack_data(dir, YP_buffer_recv, yp_buffersize, &mygrid,
				mympirank);
	}

	MPI_Barrier(MPI_COMM_WORLD );
	//Direction dir = YPdir;
	dir = YPdir;
	if (dir != DIR_NONE) {
		datapro.pack_data(dir, YP_buffer_send, yp_buffersize, &mygrid,
				mympirank);
		comm.coarray_send_interior_states(mympirank, neigh_yp, dir,
				YP_buffer_send, yp_buffersize);
		comm.coarray_recv_interior_states(neigh_yn, mympirank, dir,
				YN_buffer_recv, yn_buffersize);
		datapro.unpack_data(dir, YN_buffer_recv, yn_buffersize, &mygrid,
				mympirank);
	}

	//cout<<" for processor "<<mympirank<<", grid before exchange :"<<endl;
	//mygrid.printallelements();

	MPI_Barrier(MPI_COMM_WORLD );
	//Direction dir = ZNdir;
	dir = ZNdir;
	if (dir != DIR_NONE) {
		datapro.pack_data(dir, ZN_buffer_send, zn_buffersize, &mygrid,
				mympirank);
		comm.coarray_send_interior_states(mympirank, neigh_zn, dir,
				ZN_buffer_send, zn_buffersize);
		comm.coarray_recv_interior_states(neigh_zp, mympirank, dir,
				ZP_buffer_recv, zp_buffersize);
		datapro.unpack_data(dir, ZP_buffer_recv, zp_buffersize, &mygrid,
				mympirank);
	}

	MPI_Barrier(MPI_COMM_WORLD );
	//Direction dir = ZPdir;
	dir = ZPdir;
	if (dir != DIR_NONE) {
		datapro.pack_data(dir, ZP_buffer_send, zp_buffersize, &mygrid,
				mympirank);
		comm.coarray_send_interior_states(mympirank, neigh_zp, dir,
				ZP_buffer_send, zp_buffersize);
		comm.coarray_recv_interior_states(neigh_zn, mympirank, dir,
				ZN_buffer_recv, zn_buffersize);
		datapro.unpack_data(dir, ZN_buffer_recv, zn_buffersize, &mygrid,
				mympirank);
	}

	//cout<<" for processor "<<mympirank<<", grid before exchange :"<<endl;
	//mygrid.printallelements();
	/*
	 */

	//cout<<" for processor "<<mympirank<<", grid before exchange :"<<endl;
	//mygrid.printallelements();
	MPI_Barrier(MPI_COMM_WORLD );
}

void ocu::CoarrayMPIComm::coarray_send_interior_states(int src_id, int dst_id,
		Direction dir, double *buffer, int buffersize) {
	MPI_Status status;
	MPI_Request request;
	//cerr<<" size of the buffer is "<<sizeof buffer <<endl;
	//cerr<<" before Bsend, buffer:"<<endl;
	//cerr<< &buffer[0]<<","<<&buffer[1]<<&buffer[buffersize-1]<<endl;
	//cerr<< buffer<<endl;

	MPI_Isend(buffer, buffersize, MPI_DOUBLE, dst_id, src_id * 10 + dir,
			MPI_COMM_WORLD, &request);
	//MPI_Isend(buffer,buffersize,MPI_DOUBLE,dst_id,src_id,MPI_COMM_WORLD, &request);
	//cout<<" send complete, start waiting"<<endl;
	//cout<<" tries to send "<< buffersize <<" double numbers to "<<dst_id<<" with tag"<<src_id*10+dir<<endl;
	//cout<<" tries to send "<< buffersize <<" double numbers to "<<dst_id<<" with tag"<<src_id<<endl;
}

void ocu::CoarrayMPIComm::coarray_recv_interior_states(int src_id, int dst_id,
		Direction dir, double *buffer, int buffersize) {
	MPI_Status status;
	//cout<<" tries to receive "<< buffersize <<" double numbers from "<<src_id<<" with tag"<<src_id*10+dir<<endl;
	MPI_Recv(buffer, buffersize, MPI_DOUBLE, src_id, src_id * 10 + dir,
			MPI_COMM_WORLD, &status);
	//MPI_Recv(buffer,buffersize,MPI_DOUBLE,src_id,src_id,MPI_COMM_WORLD,&status);
	//cout<<" processor "<< dst_id<<"recv completed"<<endl;
}

void ocu::CoarrayMPIManager::calculateBufferSize() {
	xn_buffersize = (ysize + 2 * buffer_depth) * (zsize + 2 * buffer_depth)
			* buffer_depth;
	xp_buffersize = xn_buffersize;
	yn_buffersize = (xsize + 2 * buffer_depth) * (zsize + 2 * buffer_depth)
			* buffer_depth;
	yp_buffersize = yn_buffersize;
	zn_buffersize = (xsize + 2 * buffer_depth) * (ysize + 2 * buffer_depth)
			* buffer_depth;
	zp_buffersize = zn_buffersize;

	//cout<<" xn buffer size = "<<xn_buffersize<<endl;
	//cout<<" yn buffer size = "<<yn_buffersize<<endl;
	//cout<<" zn buffer size = "<<zn_buffersize<<endl;
	//cout<<" xp buffer size = "<<xn_buffersize<<endl;
	//cout<<" yp buffer size = "<<yn_buffersize<<endl;
	//cout<<" zp buffer size = "<<zn_buffersize<<endl;

}

//void  ocu::CoarrayMPIdataprocessor::pack_data(Direction dir, double* storage, int buffersize,Grid3DHostD* ori_grid, int procrank)
void ocu::CoarrayMPIdataprocessor::pack_data(Direction dir, double* storage,
		int buffersize, Grid3DTypedBase<double>* ori_grid, int procrank) {
	int count = 0;
	int i, j, k;
	if (dir == XNdir) {
		for (i = 0; i < ori_grid->gx(); i++) {
			for (j = -ori_grid->gy(); j < ori_grid->ny() + ori_grid->gy();
					j++) {
				//cout<<" printing:"<<storage[count]<<endl;
				//cout<<"while packing, printing:"<<ori_grid->at(i,j,0)<<endl;
				//cout<<" while pakcing, proc: "<<procrank<<"pos: ["<<i<<","<<j<<","<<0<<"], data to be packed: "<<ori_grid->at(i,j,k)<<endl;
				for (k = -ori_grid->gz(); k < ori_grid->nz() + ori_grid->gz();
						k++) {
					storage[count] = ori_grid->at(i, j, k);
					//cout<<" packing : in proc "<<procrank<<" ,"<<count <<" th storage space:"<<storage[count]<<endl;
					count++;
					//assert(count <= buffersize);
				}
			}
		}
	} else if (dir == XPdir) {
		//cout<<" PROC "<<procrank<<" begins packing :"<<endl;
		for (i = ori_grid->nx() - ori_grid->gx(); i < ori_grid->nx(); i++) {
			for (j = -ori_grid->gy(); j < ori_grid->ny() + ori_grid->gy();
					j++) {
				for (k = -ori_grid->gz(); k < ori_grid->nz() + ori_grid->gz();
						k++) {
					storage[count] = ori_grid->at(i, j, k);
					//cout<<" packing : in proc "<<procrank<<" ,"<<count <<" th storage space:"<<storage[count]<<endl;
					count++;
				}
			}
		}
	} else if (dir == YNdir) {
		for (i = -ori_grid->gx(); i < ori_grid->nx() + ori_grid->gx(); i++) {
			for (j = 0; j < ori_grid->gy(); j++) {
				for (k = -ori_grid->gz(); k < ori_grid->nz() + ori_grid->gz();
						k++) {
					storage[count] = ori_grid->at(i, j, k);
					count++;
				}
			}
		}
	} else if (dir == YPdir) {
		for (i = -ori_grid->gx(); i < ori_grid->nx() + ori_grid->gx(); i++) {
			for (j = ori_grid->ny() - ori_grid->gy(); j < ori_grid->ny(); j++) {
				for (k = -ori_grid->gz(); k < ori_grid->nz() + ori_grid->gz();
						k++) {
					storage[count] = ori_grid->at(i, j, k);
					count++;
				}
			}
		}
	} else if (dir == ZNdir) {
		for (i = -ori_grid->gx(); i < ori_grid->nx() + ori_grid->gx(); i++) {
			for (j = -ori_grid->gy(); j < ori_grid->ny() + ori_grid->gy();
					j++) {
				for (k = 0; k < ori_grid->gz(); k++) {
					storage[count] = ori_grid->at(i, j, k);
					//cout<<"processor"<<procrank<<" send"<<storage[count]<<endl;
					count++;
				}
			}
		}
	} else if (dir == ZPdir) {
		for (i = -ori_grid->gx(); i < ori_grid->nx() + ori_grid->gx(); i++) {
			for (j = -ori_grid->gy(); j < ori_grid->ny() + ori_grid->gy();
					j++) {
				for (k = ori_grid->nz() - ori_grid->gz(); k < ori_grid->nz();
						k++) {
					storage[count] = ori_grid->at(i, j, k);
					//cout<<"processor"<<procrank<<" send"<<storage[count]<<endl;
					count++;
				}
			}
		}
	}
}

void ocu::CoarrayMPIdataprocessor::unpack_data(Direction dir, double* storage,
		int buffersize, Grid3DTypedBase<double>* upd_grid, int procrank) {
	int count = 0;
	int i, j, k;
	if (dir == XNdir) {
		for (i = upd_grid->nx(); i < upd_grid->nx() + upd_grid->gx(); i++) {
			for (j = -upd_grid->gy(); j < upd_grid->ny() + upd_grid->gy();
					j++) {
				//cout<<" while unpakcing, proc: "<<procrank<<"pos: ["<<i<<","<<j<<","<<0<<"], data stored: "<<upd_grid->at(i,j,k)<<endl;
				for (k = -upd_grid->gz(); k < upd_grid->nz() + upd_grid->gz();
						k++) {
					upd_grid->at(i, j, k) = storage[count];
					//cout<<"unpakcing : in proc "<<procrank<<" ,"<<count<<" th storage space:"<<storage[count]<<endl;
					count++;
					//assert(count == buffersize);
				}
			}
		}
	} else if (dir == XPdir) {
		for (i = -upd_grid->gx(); i < 0; i++) {
			for (j = -upd_grid->gy(); j < upd_grid->ny() + upd_grid->gy();
					j++) {
				for (k = -upd_grid->gz(); k < upd_grid->nz() + upd_grid->gz();
						k++) {
					upd_grid->at(i, j, k) = storage[count];
					//cout<<"unpakcing : in proc "<<procrank<<" ,"<<count<<" th storage space:"<<storage[count]<<endl;
					count++;
				}
			}
		}
	}
	if (dir == YNdir) {
		for (i = -upd_grid->gx(); i < upd_grid->nx() + upd_grid->gx(); i++) {
			for (j = upd_grid->ny(); j < upd_grid->ny() + upd_grid->gy(); j++) {
				for (k = -upd_grid->gz(); k < upd_grid->nz() + upd_grid->gz();
						k++) {
					upd_grid->at(i, j, k) = storage[count];
					//cout<<"processor"<<procrank<<" receive "<<storage[count]<<" at loc:["<<i<<","<<j<<","<<k<<"]"<<endl;
					count++;
				}
			}
		}
	} else if (dir == YPdir) {
		for (i = -upd_grid->gx(); i < upd_grid->nx() + upd_grid->gx(); i++) {
			for (j = -upd_grid->gy(); j < 0; j++) {
				for (k = -upd_grid->gz(); k < upd_grid->nz() + upd_grid->gz();
						k++) {
					upd_grid->at(i, j, k) = storage[count];
					//cout<<"processor"<<procrank<<" receive "<<storage[count]<<" at loc:["<<i<<","<<j<<","<<k<<"]"<<endl;
					count++;
				}
			}
		}
	}
	if (dir == ZNdir) {
		for (i = -upd_grid->gx(); i < upd_grid->nx() + upd_grid->gx(); i++) {
			for (j = -upd_grid->gy(); j < upd_grid->ny() + upd_grid->gy();
					j++) {
				for (k = upd_grid->nz(); k < upd_grid->nz() + upd_grid->gz();
						k++) {
					//cout<<" updating ZN"<<endl;
					upd_grid->at(i, j, k) = storage[count];
					//cout<<"processor"<<procrank<<" receive "<<storage[count]<<" at loc:["<<i<<","<<j<<","<<k<<"]"<<endl;
					count++;
				}
			}
		}
	} else if (dir == ZPdir) {
		for (i = -upd_grid->gx(); i < upd_grid->nx() + upd_grid->gx(); i++) {
			for (j = -upd_grid->gy(); j < upd_grid->ny() + upd_grid->gy();
					j++) {
				for (k = -upd_grid->gz(); k < 0; k++) {
					upd_grid->at(i, j, k) = storage[count];
					//cout<<"processor"<<procrank<<" receive "<<storage[count]<<" at loc:["<<i<<","<<j<<","<<k<<"]"<<endl;
					count++;
				}
			}
		}
	}
}

