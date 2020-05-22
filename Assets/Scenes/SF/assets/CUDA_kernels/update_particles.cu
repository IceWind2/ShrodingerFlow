extern "C" __global__ void
update_particles(float* x, float* y, float* z,
                 const float* k1x, const float* k1y, const float* k1z,
                 const float* k2x, const float* k2y, const float* k2z,
                 const float* k3x, const float* k3y, const float* k3z,
                 const float* k4x, const float* k4y, const float* k4z,
                 float dt)
{
    int i = blockDim.x * blockIdx.x;
    switch (threadIdx.x)
    {
        case 0:
            x[i] += (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) * dt / 6;
            break;
        case 1:
            y[i] += (k1y[i] + 2 * k2y[i] + 2 * k3y[i] + k4y[i]) * dt / 6;
            break;
        case 2:
            z[i] += (k1z[i] + 2 * k2z[i] + 2 * k3z[i] + k4z[i]) * dt / 6;
            break;
    }
}