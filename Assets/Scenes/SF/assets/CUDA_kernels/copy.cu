extern "C" __global__ void
copy(float2* x, float* y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    x[i].x = y[i];
    x[i].y = 0;
}