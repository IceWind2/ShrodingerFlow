extern "C" __global__ void
fft_norm(float2* arr, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    arr[i].x /= size;
    arr[i].y /= size;
}