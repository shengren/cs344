/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"
#include "reference.cpp"

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
                     unsigned int* const out_val,
                     const unsigned int* const index,
                     const size_t num) {
  const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx >= num)
    return;

  unsigned int val = in_val[tidx];
  size_t new_index = index[tidx];
  out_val[new_index] = val;
}

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (gidx >= numVals)
    return;

  unsigned int a = vals[gidx];
  unsigned int b;
  if (gidx < numVals - 1)
    b = vals[gidx + 1];
  if (gidx == numVals - 1) {
    atomicAdd(&histo[a], gidx + 1);
  } else if (a != b) {
    atomicAdd(&histo[a], gidx + 1);
    atomicAdd(&histo[b], -(gidx + 1));
  }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  unsigned int *d_keys_ping;
  unsigned int *d_keys_pong;
  checkCudaErrors(cudaMalloc(&d_keys_ping, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_keys_pong, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemcpy(d_keys_ping, d_vals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

  unsigned int *d_scan_ping;
  unsigned int *d_scan_pong;
  unsigned int *d_index;
  checkCudaErrors(cudaMalloc(&d_scan_ping, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_scan_pong, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_index, sizeof(unsigned int) * numElems));

  const dim3 blockSize(1024);
  const dim3 gridSize((numElems + 1023) / 1024);
  for (unsigned int bit = 0; bit < 32; ++bit) {
    // 0
    Predicate<<<gridSize, blockSize>>>(d_keys_ping, 
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

    ComputeNewIndex<<<gridSize, blockSize>>>(d_keys_ping, 
                                             numElems, 
                                             bit, 
                                             0, 
                                             0, 
                                             d_scan_ping, 
                                             d_index);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // 1
    Predicate<<<gridSize, blockSize>>>(d_keys_ping, 
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

    ComputeNewIndex<<<gridSize, blockSize>>>(d_keys_ping, 
                                             numElems, 
                                             bit, 
                                             (1 << bit), 
                                             h_numZeros, 
                                             d_scan_ping, 
                                             d_index);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Move
    Move<<<gridSize, blockSize>>>(d_keys_ping,
                                  d_keys_pong,
                                  d_index,
                                  numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Swap input and output pointers
    unsigned int *t = d_keys_ping;
    d_keys_ping = d_keys_pong;
    d_keys_pong = t;
  }

  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  yourHisto<<<gridSize, blockSize>>>(d_keys_ping, d_histo, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(d_keys_ping));
  checkCudaErrors(cudaFree(d_keys_pong));
  checkCudaErrors(cudaFree(d_scan_ping));
  checkCudaErrors(cudaFree(d_scan_pong));
  checkCudaErrors(cudaFree(d_index));

  //Again we have provided a reference calculation for you
  //to help with debugging.  Uncomment the code below
  //to activate it.
  //REMEMBER TO COMMENT IT OUT BEFORE GRADING
  //otherwise your code will be too slow

  /*
  unsigned int *h_vals = new unsigned int[numElems];
  unsigned int *h_histo = new unsigned int[numBins];

  checkCudaErrors(cudaMemcpy(h_vals, d_vals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));

  reference_calculation(h_vals, h_histo, numBins, numElems);

  unsigned int *your_histo = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(your_histo, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  checkResultsExact(h_histo, your_histo, numBins); 

  delete[] h_vals;
  delete[] h_histo;
  delete[] your_histo;
  */
}
