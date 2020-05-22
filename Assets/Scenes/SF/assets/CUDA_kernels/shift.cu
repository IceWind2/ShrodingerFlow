extern "C" __global__ void
shift(float* arr, int resx, int resy, int resz, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = resx / 2;
    
    if (i / resy / resz < idx) {
        if (i / resz % resz < idx) {
            if (i % resz < idx) {
                //0
                arr[i] = arr[(i + idx + idx * resz + idx * resy * resz) % size];
            }
            else {
                //1
                arr[i] = arr[(i - idx + idx * resz + idx * resy * resz + size) % size];
            }
        }
        else {
            if (i % resz < idx) {
                //2
                arr[i] = arr[(i + idx - idx * resz + idx * resy * resz + size) % size];
            }
            else {
                //3
                arr[i] = arr[(i - idx - idx * resz + idx * resy * resz + size) % size];
            }
        }
    }
    else {
        if (i / resz % resz < idx) {
            if (i % resz < idx) {
                //4
                arr[i] = arr[(i + idx + idx * resz - idx * resy * resz + size) % size];
            }
            else {
                //5
                arr[i] = arr[(i - idx + idx * resz - idx * resy * resz + size) % size];
            }
        }
        else {
            if (i % resz < idx) {
                //6
                arr[i] = arr[(i + idx - idx * resz - idx * resy * resz + size) % size];
            }
            else {
                //7
                arr[i] = arr[(i - idx - idx * resz - idx * resy * resz + size) % size];
            }
        }
    }
}