//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>
#include <cstdio>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void Predicate(const unsigned int* const key,
                          const size_t num,
                          const unsigned int bit,
                          const unsigned int value,
                          unsigned int* const predicate) {
  const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  if ((key[tidx] & (1 << bit)) == value)
    predicate[tidx] = 1;
  else
    predicate[tidx] = 0;
}

__global__ void ScanStep(const unsigned int* const in,
                         unsigned int* const out,
                         const size_t num,
                         const size_t width) {
  const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  unsigned int s = in[tidx];
  if (tidx >= width)
    s += in[tidx - width];
  out[tidx] = s;
}

__global__ void ComputeNewIndex(const unsigned int* const key,
                                const size_t num,
                                const unsigned int bit,
                                const unsigned int value,
                                const unsigned int base,
                                const unsigned int* const offset,
                                unsigned int* const index) {
  const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  if ((key[tidx] & (1 << bit)) == value)
    index[tidx] = base + offset[tidx] - 1;  // inclusive -> exclusive
}

__global__ void Move(const unsigned int* const in_val,
                     const unsigned int* const in_pos,
                     unsigned int* const out_val,
                     unsigned int* const out_pos,
                     const unsigned int* const index,
                     const size_t num) {
  const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  unsigned int val = in_val[tidx];
  unsigned int pos = in_pos[tidx];
  size_t new_index = index[tidx];
  out_val[new_index] = val;
  out_pos[new_index] = pos;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code MUST RUN BEFORE YOUR CODE in case you accidentally change       *
  * the input values when implementing your radix sort.                       *
  *                                                                           *
  * This code performs the reference radix sort on the host and compares your *
  * sorted values to the reference.                                           *
  *                                                                           *
  * Thrust containers are used for copying memory from the GPU                *
  * ************************************************************************* */
  
  /*
  thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(d_inputVals),
                                                thrust::device_ptr<unsigned int>(d_inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(d_inputPos),
                                               thrust::device_ptr<unsigned int>(d_inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
                        &h_outputVals[0], &h_outputPos[0],
                        numElems);
  */

  //PUT YOUR SORT HERE
  unsigned int *d_scan_ping;
  checkCudaErrors(cudaMalloc(&d_scan_ping, sizeof(unsigned int) * numElems));
  unsigned int *d_scan_pong;
  checkCudaErrors(cudaMalloc(&d_scan_pong, sizeof(unsigned int) * numElems));
  unsigned int *d_index;
  checkCudaErrors(cudaMalloc(&d_index, sizeof(unsigned int) * numElems));

  const dim3 blockSize(1024);
  const dim3 gridSize((numElems + 1023) / 1024);

  unsigned int *d_in_val = d_inputVals;
  unsigned int *d_in_pos = d_inputPos;
  unsigned int *d_out_val = d_outputVals;
  unsigned int *d_out_pos = d_outputPos;
  for (unsigned int bit = 0; bit < 32; ++bit) {
    // 0
    Predicate<<<gridSize, blockSize>>>(d_in_val, 
                                       numElems, 
                                       bit, 
                                       0, 
                                       d_scan_ping);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for (unsigned int width = 1; width < numElems; width <<= 1) {
      ScanStep<<<gridSize, blockSize>>>(d_scan_ping, 
                                        d_scan_pong, 
                                        numElems, 
                                        width);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      // Swap. d_scan_ping has the final scan results.
      unsigned int *t = d_scan_ping;
      d_scan_ping = d_scan_pong;
      d_scan_pong = t;
    }

    unsigned int h_numZeros;
    checkCudaErrors(cudaMemcpy(&h_numZeros, 
                               &d_scan_ping[numElems - 1], 
                               sizeof(unsigned int), 
                               cudaMemcpyDeviceToHost));
    //printf("#0s = %d\n", h_numZeros);

    ComputeNewIndex<<<gridSize, blockSize>>>(d_in_val, 
                                             numElems, 
                                             bit, 
                                             0, 
                                             0, 
                                             d_scan_ping, 
                                             d_index);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // 1
    Predicate<<<gridSize, blockSize>>>(d_in_val, 
                                       numElems, 
                                       bit, 
                                       (1 << bit), 
                                       d_scan_ping);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for (unsigned int width = 1; width < numElems; width <<= 1) {
      ScanStep<<<gridSize, blockSize>>>(d_scan_ping, 
                                        d_scan_pong, 
                                        numElems, 
                                        width);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      // Swap. d_scan_ping has the final scan results.
      unsigned int *t = d_scan_ping;
      d_scan_ping = d_scan_pong;
      d_scan_pong = t;
    }
    /*
    unsigned int h_numOnes;
    checkCudaErrors(cudaMemcpy(&h_numOnes, &d_scan_ping[numElems - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("#1s = %d\n", h_numOnes);
    printf("total = %d(%d)\n", numElems, h_numZeros + h_numOnes);
    */

    ComputeNewIndex<<<gridSize, blockSize>>>(d_in_val, 
                                             numElems, 
                                             bit, 
                                             (1 << bit), 
                                             h_numZeros, 
                                             d_scan_ping, 
                                             d_index);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Move
    Move<<<gridSize, blockSize>>>(d_in_val, d_in_pos, d_out_val, d_out_pos, 
                                  d_index, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Swap input and output pointers
    unsigned int *t = d_in_val;
    d_in_val = d_out_val;
    d_out_val = t;
    t = d_in_pos;
    d_in_pos = d_out_pos;
    d_out_pos = t;
  }
  // After 32 iterations, results are in d_output*.
  checkCudaErrors(cudaMemcpy(d_outputVals, 
                             d_inputVals, 
                             sizeof(unsigned int) * numElems, 
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, 
                             d_inputPos, 
                             sizeof(unsigned int) * numElems, 
                             cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaFree(d_scan_ping));
  checkCudaErrors(cudaFree(d_scan_pong));
  checkCudaErrors(cudaFree(d_index));

  /* *********************************************************************** *
   * Uncomment the code below to do the correctness checking between your    *
   * result and the reference.                                               *
   **************************************************************************/

  /*
  thrust::host_vector<unsigned int> h_yourOutputVals(thrust::device_ptr<unsigned int>(d_outputVals),
                                                     thrust::device_ptr<unsigned int>(d_outputVals) + numElems);
  thrust::host_vector<unsigned int> h_yourOutputPos(thrust::device_ptr<unsigned int>(d_outputPos),
                                                    thrust::device_ptr<unsigned int>(d_outputPos) + numElems);

  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);
  */
}
