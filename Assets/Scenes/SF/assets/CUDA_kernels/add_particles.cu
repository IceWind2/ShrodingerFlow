extern "C" __global__ void
add_particles(float* x, float* y, float* z, const float* xx, const float* yy, const float* zz, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    x[i + size] = xx[i];
    y[i + size] = yy[i];
    z[i + size] = zz[i];
}