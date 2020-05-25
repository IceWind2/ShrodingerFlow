extern "C" __global__ void
mul_each(float2* x, float2* y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    x[i] = make_float2(x[i].x * y[i].x - x[i].y * y[i].y, x[i].x * y[i].y + x[i].y * y[i].x);
}