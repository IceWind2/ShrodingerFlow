extern "C" __global__ void
staggered_sharp(float* arr, float d)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    arr[i] = arr[i] / d;
}