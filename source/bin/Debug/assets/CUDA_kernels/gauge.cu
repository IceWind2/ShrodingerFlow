extern "C" __global__ void
gauge(float2* psi1, float2* psi2, float2* q, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float2 eiq = make_float2(cos(-q[i].x / size), sin(-q[i].x / size));
    
    psi1[i] = make_float2(psi1[i].x * eiq.x - psi1[i].y * eiq.y, psi1[i].x * eiq.y + psi1[i].y * eiq.x);
    psi2[i] = make_float2(psi2[i].x * eiq.x - psi2[i].y * eiq.y, psi2[i].x * eiq.y + psi2[i].y * eiq.x);
}