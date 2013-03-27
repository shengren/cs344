#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "compare.h"
#include "gputimer.h"

// http://en.wikipedia.org/wiki/Bitonic_sort
__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
    // you are guaranteed this is called with <<<1, 64, 64*4>>>
    extern __shared__ float sdata[];
    int tid  = threadIdx.x;
    sdata[tid] = d_in[tid];
    __syncthreads();
    
    for (int stage = 0; stage <= 5; stage++)
    {
      int s = 1 << stage;
      int i = (tid / s) * s * 2 + tid % s;
      bool d = (tid / s) % 2 == 0;
      for (int substage = stage; substage >= 0; substage--)
      {
        if (tid < 32) {
          int p = 1 << substage;
          if (d) {
            if (sdata[i] > sdata[i + p]) {
              float t = sdata[i];
              sdata[i] = sdata[i + p];
              sdata[i + p] = t;
            }
          } else {
            if (sdata[i] < sdata[i + p]) {
              float t = sdata[i];
              sdata[i] = sdata[i + p];
              sdata[i + p] = t;
            }
          }
          int np = 1 << (substage - 1);
          if ((tid / np) % 2 == 1)
            i += np;
        }
        __syncthreads();
      }
    }

    d_out[tid] = sdata[tid];
}

int compareFloat (const void * a, const void * b)
{
  if ( *(float*)a <  *(float*)b ) return -1;
  if ( *(float*)a == *(float*)b ) return 0;
  if ( *(float*)a >  *(float*)b ) return 1;
  return 0;                     // should never reach this
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float h_sorted[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float)random()/(float)RAND_MAX;
        h_sorted[i] = h_in[i];
    }
    qsort(h_sorted, ARRAY_SIZE, sizeof(float), compareFloat);

    // declare GPU memory pointers
    float * d_in, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // launch the kernel
    GpuTimer timer;
    timer.Start();
    batcherBitonicMergesort64<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(float)>>>(d_out, d_in);
    timer.Stop();
    
    printf("Your code executed in %g ms\n", timer.Elapsed());
    
    // copy back the sum from GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    compare(h_out, h_sorted, ARRAY_SIZE);
  
    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
        
    return 0;
}
