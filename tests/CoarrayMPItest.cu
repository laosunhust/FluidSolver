#include "tests/testframework.h"
#include "ocustorage/coarraympi.h"
#include "ocustorage/coarray.h"

using namespace ocu;

DECLARE_UNITTEST_MULTIGPU_BEGIN(MultiReduceTest);

void run()
{
    int argc = 1;
    char *argv[] = {"aa"};
    //initMPIEnvironment(argc, argv);
    //MPI reduce: MIN
    //MPI reduce: MAX
    //MPI reduce: SUM
    //MPI reduce: SQRSUM
    //cout<<"Number of images:"<<ThreadManager::num_images()<<endl;
}

DECLARE_UNITTEST_END(MultiReduceTest);

DECLARE_UNITTEST_MPIGPU_BEGIN(CoArrayMPI1DTest);

void run()
{
	cout<<"MPI test running!"<<endl;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
	MPI_Comm_size(MPI_COMM_WORLD,  &size);
	ocu::CoarrayMPIManager mpiManager(rank,size);
	mpiManager.printMyInfo();
	mpiManager.dataExchangeWithNeighbors();
    //First init with same para
    //assert equal on CPU side
    //assert equal on GPU side
    //Then both perform some computation
    //Sync
    //assert equal on CPU side
    //assert equal on GPU side

}

DECLARE_UNITTEST_END(CoArrayMPI1DTest);

DECLARE_UNITTEST_MULTIGPU_BEGIN(CoArrayMPI3DTest);

void run()
{
  //First init with same para
  //assert equal on CPU side
  //assert equal on GPU side
  //Then both perform some computation
  //Sync
  //assert equal on CPU side
  //assert equal on GPU side  
}

DECLARE_UNITTEST_END(CoArrayMPI3DTest);
