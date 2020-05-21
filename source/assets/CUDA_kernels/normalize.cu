extern "C" __global__ void
normalize(float2* psi1, float2* psi2)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    float sqAbs1 = psi1[i].x * psi1[i].x + psi1[i].y * psi1[i].y;
    float sqAbs2 = psi2[i].x * psi2[i].x + psi2[i].y * psi2[i].y;
    
    float norm = sqrt(sqAbs1 + sqAbs2);
    
    psi1[i].x /= norm;
    psi1[i].y /= norm;
    psi2[i].x /= norm;
    psi2[i].y /= norm;
}