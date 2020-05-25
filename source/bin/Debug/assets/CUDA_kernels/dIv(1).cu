extern "C" __global__ void
staggered_sharp(float* res, float* vx, float* vy, float* vz, float dx, float dy, float dz, int res, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    res[i] = (vx[i] - vx[(i - 2 * res * res + size) % size]) / dx / dx;
    res[i] = res[i] + (vy[i] - vy[(i - 2 * res + size) % size]) / dy / dy;
    res[i] = res[i] + (vz[i] - vz[(i - 2 + size) % size]) / dz / dz;
}