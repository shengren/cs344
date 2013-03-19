//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

__global__ void Test(unsigned char *m,
                     size_t nr, size_t nc,
                     uchar4 *img) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;
  uchar4 p;
  if (m[i] == 0)
    p.x = p.y = p.z = 0;
  else if (m[i] == 1)
    p.x = p.y = p.z = 255;
  else if (m[i] == 2)
    p.x = p.y = p.z = 127;
  img[i] = p;
}

__global__ void CreateMask(uchar4 *img, 
                           size_t nr, size_t nc, 
                           unsigned char *m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;
  uchar4 p = img[i];
  if (p.x + p.y + p.z < 3 * 255)
    m[i] = 1;
  else
    m[i] = 0;
}

__global__ void CreateMask2(uchar4 *img, 
                            size_t nr, size_t nc, 
                            unsigned char *m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;
  if (m[i] > 0 &&
      m[i - 1] > 0 &&
      m[i + 1] > 0 &&
      m[i - nc] > 0 &&
      m[i + nc] > 0)
    m[i] = 2;
}

__global__ void SeparateChannels(uchar4 *img, 
                                 size_t nr, size_t nc,
                                 unsigned char *r,
                                 unsigned char *g,
                                 unsigned char *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;
  uchar4 p = img[i];
  r[i] = p.x;
  g[i] = p.y;
  b[i] = p.z;
}

__global__ void ComputeG(unsigned char *c, 
                         unsigned char *m,
                         size_t nr, size_t nc,
                         float *g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;

  if (m[i] != 2) {
    g[i] = 0.0f;
    return;
  }

  float sum = 4.0f * (float)c[i];
  sum -= (float)c[i - 1];
  sum -= (float)c[i + 1];
  sum -= (float)c[i - nc];
  sum -= (float)c[i + nc];
  g[i] = sum;
}

__global__ void Initialize(unsigned char *src, 
                           unsigned char *dest, 
                           unsigned char *m,
                           size_t nr, size_t nc,
                           float *init) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;
  unsigned char c;
  if (m[i] == 2)
    c = src[i];
  else
    c = dest[i];
  init[i] = (float)c;
  //init[i] = (float)src[i];
}

__global__ void Iterate(unsigned char *m, 
                        unsigned char *src,
                        unsigned char *dest,
                        float *ping, 
                        float *g,
                        size_t nr, size_t nc,
                        float *pong) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;

  if (m[i] != 2) {
    pong[i] = ping[i];
    //pong[i] = src[i];
    return;
  }

  float sum = 0.0f;
  if (m[i - 1] == 2) sum += ping[i - 1];
  else sum += (float)dest[i - 1];
  if (m[i + 1] == 2) sum += ping[i + 1];
  else sum += (float)dest[i + 1];
  if (m[i - nc] == 2) sum += ping[i - nc];
  else sum += (float)dest[i - nc];
  if (m[i + nc] == 2) sum += ping[i + nc];
  else sum += (float)dest[i + nc];
  sum += g[i];
  pong[i] = min(255.0f, max(0.0f, sum / 4.0f));
}

__global__ void WriteResult(float *p, size_t nr, size_t nc, size_t ci, uchar4 *res) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * nc)
    return;
  if (ci == 0) res[i].x = (unsigned char)p[i];
  else if (ci == 1) res[i].y = (unsigned char)p[i];
  else if (ci == 2) res[i].z = (unsigned char)p[i];
}

#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /*
     To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

  size_t srcSize = numRowsSource * numColsSource;
  dim3 gridSize((srcSize + 1023) / 1024);
  dim3 blockSize(1024);

  uchar4 *d_sourceImg = NULL;
  checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg,
                             sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));
  uchar4 *d_destImg = NULL;
  checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg,
                             sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));
  unsigned char *d_mask = NULL;
  checkCudaErrors(cudaMalloc(&d_mask, sizeof(unsigned char) * srcSize));
  unsigned char *d_srcRed = NULL;
  unsigned char *d_srcGreen = NULL;
  unsigned char *d_srcBlue = NULL;
  checkCudaErrors(cudaMalloc(&d_srcRed, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_srcGreen, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_srcBlue, sizeof(unsigned char) * srcSize));
  unsigned char *d_destRed = NULL;
  unsigned char *d_destGreen = NULL;
  unsigned char *d_destBlue = NULL;
  checkCudaErrors(cudaMalloc(&d_destRed, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_destGreen, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_destBlue, sizeof(unsigned char) * srcSize));
  float *d_ping = NULL;
  float *d_pong = NULL;
  checkCudaErrors(cudaMalloc(&d_ping, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_pong, sizeof(float) * srcSize));
  float *d_g = NULL;
  checkCudaErrors(cudaMalloc(&d_g, sizeof(float) * srcSize));
  uchar4 *d_blendedImg = NULL;
  checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * srcSize));

  CreateMask<<<gridSize, blockSize>>>(d_sourceImg,
                                      numRowsSource, numColsSource,
                                      d_mask);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  CreateMask2<<<gridSize, blockSize>>>(d_sourceImg,
                                       numRowsSource, numColsSource,
                                       d_mask);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  SeparateChannels<<<gridSize, blockSize>>>(d_sourceImg,
                                            numRowsSource, numColsSource,
                                            d_srcRed, d_srcGreen, d_srcBlue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  SeparateChannels<<<gridSize, blockSize>>>(d_destImg,
                                            numRowsSource, numColsSource,
                                            d_destRed, d_destGreen, d_destBlue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Red
  ComputeG<<<gridSize, blockSize>>>(d_srcRed, 
                                    d_mask,
                                    numRowsSource, numColsSource,
                                    d_g);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  Initialize<<<gridSize, blockSize>>>(d_srcRed,
                                      d_destRed,
                                      d_mask,
                                      numRowsSource, numColsSource,
                                      d_ping);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  for (size_t k = 0; k < 800; ++k) {
    Iterate<<<gridSize, blockSize>>>(d_mask,
                                     d_srcRed,
                                     d_destRed,
                                     d_ping,
                                     d_g,
                                     numRowsSource, numColsSource,
                                     d_pong);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::swap(d_ping, d_pong);
  }
  WriteResult<<<gridSize, blockSize>>>(d_ping, numRowsSource, numColsSource, 0, d_blendedImg);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Green
  ComputeG<<<gridSize, blockSize>>>(d_srcGreen, 
                                    d_mask,
                                    numRowsSource, numColsSource,
                                    d_g);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  Initialize<<<gridSize, blockSize>>>(d_srcGreen,
                                      d_destGreen, 
                                      d_mask,
                                      numRowsSource, numColsSource,
                                      d_ping);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  for (size_t k = 0; k < 800; ++k) {
    Iterate<<<gridSize, blockSize>>>(d_mask,
                                     d_srcGreen,
                                     d_destGreen, 
                                     d_ping, 
                                     d_g,
                                     numRowsSource, numColsSource,
                                     d_pong);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::swap(d_ping, d_pong);
  }
  WriteResult<<<gridSize, blockSize>>>(d_ping, numRowsSource, numColsSource, 1, d_blendedImg);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Blue
  ComputeG<<<gridSize, blockSize>>>(d_srcBlue, 
                                    d_mask,
                                    numRowsSource, numColsSource,
                                    d_g);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  Initialize<<<gridSize, blockSize>>>(d_srcBlue, 
                                      d_destBlue, 
                                      d_mask,
                                      numRowsSource, numColsSource,
                                      d_ping);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  for (size_t k = 0; k < 800; ++k) {
    Iterate<<<gridSize, blockSize>>>(d_mask,
                                     d_srcBlue,
                                     d_destBlue, 
                                     d_ping, 
                                     d_g,
                                     numRowsSource, numColsSource,
                                     d_pong);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::swap(d_ping, d_pong);
  }
  WriteResult<<<gridSize, blockSize>>>(d_ping, numRowsSource, numColsSource, 2, d_blendedImg);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * srcSize,
                             cudaMemcpyDeviceToHost));

  /* 
     The reference calculation is provided below, feel free to use it
     for debugging purposes. 
  */

  /*
  uchar4* h_reference = new uchar4[srcSize];
  reference_calc(h_sourceImg, numRowsSource, numColsSource, h_destImg, h_reference);
  memcpy(h_blendedImg, h_reference, sizeof(uchar4) * srcSize);
  */

  /*
    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference;
  */
}
