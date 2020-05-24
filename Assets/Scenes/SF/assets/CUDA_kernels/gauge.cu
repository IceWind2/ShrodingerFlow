extern "C" __global__ void
gauge(float2* psi1, float2* psi2, float2* q)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float2 eiq = make_float2(exp(q[i].y) * cos(-q[i].x), exp(q[i].y) * sin(-q[i].x));
    
    psi1[i] = make_float2(psi1[i].x * eiq.x - psi1[i].y * eiq.y, psi1[i].x * eiq.y + psi1[i].y * eiq.x);
    psi2[i] = make_float2(psi2[i].x * eiq.x - psi2[i].y * eiq.y, psi2[i].x * eiq.y + psi2[i].y * eiq.x);
    
    if (!isfinite(psi1[i].x)) {
        psi1[i].x = -1;
        psi1[i].y = -1;
    }
    if (!isfinite(psi2[i].x)) {
        psi2[i].x = -1;
        psi2[i].y = -1;
    }
}