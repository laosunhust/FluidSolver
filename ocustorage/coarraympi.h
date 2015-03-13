#ifndef _COARRAY_MPI_
#define _COARRAY_MPI_
#include<vector>
#include<iostream>
#include<assert.h>
#include<mpi.h>
#include"ocustorage/grid3dboundary.h" 
#include"grid3d.h"
#include <map>

using namespace std;
namespace ocu {
class Partition_strategy {
	int partition_type;
	int xdiv, ydiv, zdiv;
public:
	Partition_strategy() {
		partition_type = 0;
		xdiv = 1;
		ydiv = 1;
		zdiv = 1;
	}
	Partition_strategy(int x_d, int y_d, int z_d) {
		partition_type = 1;
		xdiv = x_d;
		ydiv = y_d;
		zdiv = z_d;
	}
	void set(int x_d, int y_d, int z_d) {
		partition_type = 1;
		xdiv = x_d;
		ydiv = y_d;
		zdiv = z_d;
	}
	int type() const {
		return partition_type;
	}

	int xd() const {
		return xdiv;
	}
	int yd() const {
		return ydiv;
	}
	int zd() const {
		return zdiv;
	}
};

struct Global_Grid {
public:
	int globalx;
	int globaly;
	int globalz;
	Global_Grid(int nx, int ny, int nz) {
		globalx = nx;
		globaly = ny;      //gridconfig = generate_config();
		globalz = nz;
	}
	;
	Global_Grid() {
		globalx = -1;
		globaly = -1;
		globalz = -1;
	}
	std::vector<int> getxyzdim(int myrank, int totalproc,
			Partition_strategy ps = Partition_strategy());
};

struct Local_grid {

};

class CoarrayMPIConfig {

	vector<int> npe; // numofpartitionineachdirection
public:
	CoarrayMPIConfig(int xn, int yn, int zn);
	void set(int xn, int yn, int zn);

	int getnpx() const {
		assert(npe.size() == 3);
		return npe[0];
	}
	int getnpy() const {
		assert(npe.size() == 3);
		return npe[1];
	}
	int getnpz() const {
		assert(npe.size() == 3);
		return npe[2];
	}

	CoarrayMPIConfig();
	bool is_valid() {
		if (npe.size() != 3) {
			return 0;
		}
		if (npe[0] >= 1 && npe[1] >= 1 && npe[2] >= 1) {
			return 1;
		} else {
			return 0;
		}
	}
};

enum Direction {
	DIR_NONE, XNdir, XPdir, YNdir, YPdir, ZNdir, ZPdir
};

class CoarrayMPIComm {
public:
	void coarray_send_interior_states(int src_id, int dst_id, Direction dir,
			double *buffer, int buffersize);
	void coarray_recv_interior_states(int src_id, int dst_id, Direction dir,
			double *buffer, int buffersize);
};

class CoarrayMPIdataprocessor {
public:
	void pack_data(Direction dir, double* storage, int buffersize,
			ocu::Grid3DTypedBase<double>* rawdata, int procrank);
	void unpack_data(Direction dir, double *storage, int buffersize,
			Grid3DTypedBase<double> *unpackeddata, int procrank);
};
/*class Pack_rules
 {
 public:
 int pnx,pny,pnz;
 };*/

/*
 * Each processor will have a coarrayMPIManager that does:
 * 1) Initialize MPI environment
 * 2) Initialize GPU data for PDE initial condition
 * 3) fetch data from GPU computation halo region
 * 4) exchange data with neighbors
 * 5) copy data back to GPU memory
 */
class CoarrayMPIManager {
	// domain in the global grid. xyz positive and negative directions
	int xn, xp, yn, yp, zn, zp;
	// rank of neighbor processors.  On boundary, neighbor is the processor itself.
	int neigh_xn, neigh_xp, neigh_yn, neigh_yp, neigh_zn, neigh_zp;
	int xn_buffersize, xp_buffersize;
	int yn_buffersize, yp_buffersize;
	int zn_buffersize, zp_buffersize;
private:
	double *XN_buffer_send, *XP_buffer_send;
	double *YN_buffer_send, *YP_buffer_send;
	double *ZN_buffer_send, *ZP_buffer_send;
	double *XN_buffer_recv, *XP_buffer_recv;
	double *YN_buffer_recv, *YP_buffer_recv;
	double *ZN_buffer_recv, *ZP_buffer_recv;

	// computational domain size
	int xsize, ysize, zsize;
	int mympirank, systemsize;
	Grid3DHostD mygrid;
	Grid3DDeviceD *deviceData;
	//Grid3DHostD          current_grid;
	int buffer_depth;

	CoarrayMPIConfig gridconfig;
	Partition_strategy ps;
	Global_Grid ggr;
	CoarrayMPIComm comm;
	CoarrayMPIdataprocessor datapro;
	//Local_Grid         lgr;
	//vector<double>     xpbuffer,xnbuffer,
	// ypbuffer,ynbuffer,zpbuffer,znbuffer;
public:
	int XN() const {
		return xn;
	}  // x negative neighbor
	int XP() const {
		return xp;
	}  // x positive neighbor
	int YN() const {
		return yn;
	}  // y negative neighbor
	int YP() const {
		return yp;
	}  // y positive neighbor
	int ZN() const {
		return zn;
	}  // z negative neighbor
	int ZP() const {
		return zp;
	}  // z positive neighbor

	/*
	 * Constructors. Need to remove some unused ones.
	 */
	CoarrayMPIManager();
	CoarrayMPIManager(const CoarrayMPIManager &init);
	CoarrayMPIManager(int myrank, int totalNumProc, Global_Grid compuGrid =
			Global_Grid(100, 50, 50), int buf_depth = 1);
	CoarrayMPIManager(int myrank, CoarrayMPIConfig &mpiconfig);
	CoarrayMPIManager(int myrank, int totalsize, Partition_strategy &pst,
			Global_Grid &glgr, int buffer_depth = 1);
	CoarrayMPIManager(int myrank, int totalsize, Partition_strategy &pst,
			Global_Grid &glgr, ocu::Grid3DDeviceD *devdata);

	~CoarrayMPIManager();

	/*
	 * These functions are used for (1) initializing MPI environment
	 * In a standard procedure, we need to
	 * 1) Generate CoarrayMPIConfig which contains information for number of partitions in each direction.
	 *    In this Class, it is called gridconfig; And step 1 corresponding to generate_config();
	 * 2) Then we need to call getxyzdim() so we will know how many grid points we need to generate
	 * 3) Next we calculate buffer size from xyz dim that we just obtained from previous step
	 * 4) We allocate memory on CPU for buffer by calling allocate_buffer();
	 * 5) Finally we need to determine the neighbors of this node by calling determine_neighbors()
	 */
	void generateMpiConfig();
	void calculateGridDimension();
	void calculateBufferSize();
	void allocateBuffer();
	bool determineNeighbors();

	/*
	 * These functions are used for (2) Initialize GPU data for PDE initial condition
	 * For now we only accept GPU data of format Grid3DDeviceD.
	 * We need to add an array of pointers to Grid3DDeviceD so that we can copy their values on
	 * the boundary to CPU memory
	 */
	void registerGpuDataPointers(std::vector<Grid3DDeviceD *>);
	void initGPUMemFromCPUMem(std::map<Grid3DHostD*, Grid3DDeviceD*> HostToDeviceMemoryMap);

	/*
	 * These functions are used for (3) fetch data from GPU computation halo region
	 */
    void copyDataFromGpu();

	/*
	 * these functions are used for (4) exchange data with neighbors
	 */
	bool dataExchangeWithNeighbors();

	/*
	 * These functions are used for (5) copy data back to GPU memory
	 */
    void copyDataBackToGpu();

	/*
	 * these functions are for debug purpose only
	 */
	void printMyInfo();
};
}

#endif  //_COARRAY_MPI_
