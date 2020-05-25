extern "C" __global__ void
shift(float2* arr, int resx, int resy, int resz, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = resx / 2;
    int idy = resy / 2;
    int idz = resz / 2;
    
    float2 tmp;
    
    if (i / resy / resz < idx) {
        if (i / resz % resy < idy) {
            if (i % resz < idz) {
                //0
                tmp = arr[i];
                arr[i] = arr[i + idz + idy * resz + idx * resy * resz];
                arr[i + idz + idy * resz + idx * resy * resz] = tmp;
            }
            else {
                //1
                tmp = arr[i];
                arr[i] = arr[i - idz + idy * resz + idx * resy * resz];
                arr[i - idz + idy * resz + idx * resy * resz] = tmp;
            }
        }
        else {
            if (i % resz < idz) {
                //2
                tmp = arr[i];
                arr[i] = arr[i + idz - idy * resz + idx * resy * resz];
                arr[i + idz - idy * resz + idx * resy * resz] = tmp;
            }
            else {
                //3
                tmp = arr[i];
                arr[i] = arr[i - idz - idy * resz + idx * resy * resz];
                arr[i - idz - idy * resz + idx * resy * resz] = tmp;
            }
        }
    }
}