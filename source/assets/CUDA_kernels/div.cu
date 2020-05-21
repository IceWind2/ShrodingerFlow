extern "C" __global__ void
div(float* result, float* vx, float* vy, float* vz, float dx, float dy, float dz, int res, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    result[i] = (vx[i] - vx[(i - 2 * res * res + size) % size]) / dx / dx;
    result[i] = result[i] + (vy[i] - vy[(i - 2 * res + size) % size]) / dy / dy;
    result[i] = result[i] + (vz[i] - vz[(i - 2 + size) % size]) / dz / dz;
}