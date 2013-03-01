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

__global__ void ExclusivePrefixSum(unsigned int* const d_histo) {
  __shared__ unsigned int bin[1024];
  
  int tidx = threadIdx.x;

  bin[tidx] = d_histo[tidx];
  __syncthreads();

  // Reduce
  for (size_t w = 2; w <= 1024; w <<= 1) {
    if (tidx % w == w - 1)
      bin[tidx] += bin[tidx - w / 2];
    __syncthreads();
  }

  // Downsweep
  if (tidx == 1023)
    bin[1023] = 0;
  __syncthreads();

  for (size_t w = 1024; w >= 2; w >>= 1) {
    if (tidx % w == w - 1) {
      unsigned int s = bin[tidx - w / 2] + bin[tidx];
      bin[tidx - w / 2] = bin[tidx];
      bin[tidx] = s;
    }
    __syncthreads();
  }

  d_histo[tidx] = bin[tidx];
}

__global__ void BuildHistogram(unsigned int* const d_val,
                               const size_t num,
                               unsigned int* const d_histo,
                               unsigned int* const d_offset,
                               unsigned int mask,
                               unsigned int shift) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  unsigned int key = d_val[tidx];
  unsigned int bin_idx = ((key >> shift) & mask);
  unsigned int offset = atomicAdd(&d_histo[bin_idx], 1);
  d_offset[tidx] = offset;
}

__global__ void Move(unsigned int* const d_in_val,
                     unsigned int* const d_in_pos,
                     unsigned int* const d_out_val,
                     unsigned int* const d_out_pos,
                     const size_t num,
                     unsigned int* const d_histo,
                     unsigned int* const d_offset,
                     unsigned int mask,
                     unsigned int shift) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  unsigned int val = d_in_val[tidx];
  unsigned int pos = d_in_pos[tidx];
  unsigned int bin_idx = ((val >> shift) & mask);
  unsigned int base = d_histo[bin_idx];
  unsigned int offset = d_offset[tidx];
  unsigned int new_idx = base + offset;
  d_out_val[new_idx] = val;
  d_out_pos[new_idx] = pos;
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
  
  thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(d_inputVals),
                                                thrust::device_ptr<unsigned int>(d_inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(d_inputPos),
                                               thrust::device_ptr<unsigned int>(d_inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
                        &h_outputVals[0], &h_outputPos[0],
                        numElems);

  //TODO
  //PUT YOUR SORT HERE

  const int numBits = 10;
  const int numBins = 1 << numBits;  // 1024
  printf("numBits = %d numBins = %d\n", numBits, numBins);
  unsigned int *d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
  unsigned int *d_offset;
  checkCudaErrors(cudaMalloc(&d_offset, sizeof(unsigned int) * numElems));
  unsigned int *d_mark;
  checkCudaErrors(cudaMalloc(&d_mark, sizeof(unsigned int) * numElems));
  const dim3 blockSize(1024);  // Max number of threads per block
  const dim3 gridSize((numElems + 1023) / 1024);
  printf("numElems = %d, numBlocks = %d, numThreads = %d\n", numElems, gridSize.x, blockSize.x);

  unsigned int *d_in_val = d_inputVals;
  unsigned int *d_in_pos = d_inputPos;
  unsigned int *d_out_val = d_outputVals;
  unsigned int *d_out_pos = d_outputPos;
  for (unsigned int i = 0; i < 32; i += numBits) {
    printf("i = %d\n", i);
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

    BuildHistogram<<<gridSize, blockSize>>>(d_in_val,
                                            numElems,
                                            d_histo,
                                            d_offset,
                                            (1 << numBits) - 1,
                                            i);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ExclusivePrefixSum<<<1, 1024>>>(d_histo);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //unsigned int *h_histo = new unsigned int[1024];
    //checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(unsigned int) * 1024, cudaMemcpyDeviceToHost));
    //for (int k = 0; k < 1024; ++k)
    //  printf("h[%d] = %d\n", k, h_histo[k]);
    for (int j = 0; j < numBins; ++j) {
    }

    Move<<<gridSize, blockSize>>>(d_in_val,
                                  d_in_pos,
                                  d_out_val,
                                  d_out_pos,
                                  numElems,
                                  d_histo,
                                  d_offset,
                                  (1 << numBits) - 1,
                                  i);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //unsigned int *h_test = new unsigned int[numElems];
    //checkCudaErrors(cudaMemcpy(h_test, d_out_val, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
    //for (int k = 0; k < 10; ++k)
    //  printf("  %d = %d\n", k, h_test[k]);
    //delete [] h_test;

    unsigned int *t = d_in_val;
    d_in_val = d_out_val;
    d_out_val = t;
    t = d_in_pos;
    d_in_pos = d_out_pos;
    d_out_pos = t;
  }
  checkCudaErrors(cudaMemcpy(d_outputVals,
                             d_inputVals,
                             sizeof(unsigned int) * numElems,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos,
                             d_inputPos,
                             sizeof(unsigned int) * numElems,
                             cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaFree(d_histo));
  checkCudaErrors(cudaFree(d_offset));
  checkCudaErrors(cudaFree(d_mark));

  /* *********************************************************************** *
   * Uncomment the code below to do the correctness checking between your    *
   * result and the reference.                                               *
   **************************************************************************/

  thrust::host_vector<unsigned int> h_yourOutputVals(thrust::device_ptr<unsigned int>(d_outputVals),
                                                     thrust::device_ptr<unsigned int>(d_outputVals) + numElems);
  thrust::host_vector<unsigned int> h_yourOutputPos(thrust::device_ptr<unsigned int>(d_outputPos),
                                                    thrust::device_ptr<unsigned int>(d_outputPos) + numElems);

  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);
}
