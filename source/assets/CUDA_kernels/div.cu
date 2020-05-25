extern "C" __global__ void
div(float* result, float* vx, float* vy, float* vz, float dx, float dy, float dz, int resx, int resy, int resz, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    result[i] = (vx[i] - vx[(i - resy * resz + size) % size]) / dx / dx;
    result[i] = result[i] + (vy[i] - vy[i - ((i / resz) % resy) * resz + ((((i - resz + size) % size) / resz) % resy) * resz]) / dy / dy;
    result[i] = result[i] + (vz[i] - vz[(i / resz) * resz + ((i - 1 + size) % size) % resz]) / dz / dz;
}