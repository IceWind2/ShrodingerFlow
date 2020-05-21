extern "C" __global__ void
velocity_one(float2* psi1, float2* psi2, int resy, int resz, int num, float hbar, float pi, float* vx, float* vy, float* vz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    float2 c1 = make_float2(psi1[i].x, -psi1[i].y);
    float2 c2 = make_float2(psi2[i].x, -psi2[i].y);
    
    float2 mul1 = psi1[(i + 1) % num];
    float2 mul2 = psi2[(i + 1) % num];
    float2 summ = make_float2((c1.x * mul1.x - c1.y * mul1.y + c2.x * mul2.x - c2.y * mul2.y), 
                             (c1.x * mul1.y + c1.y * mul1.x + c2.x * mul2.y + c2.y * mul2.x));
    /*float result = 0;
    if (summ.x > 0) {
        result = atan(summ.y, summ.x);
    }
    else if (summ.y > 0) {
        result = pi - atan(summ.y, -summ.x);
    }
    else {
        result = -pi + atan(summ.y, summ.x);
    }
    vz[i] = (float)result * hbar;
                            
    mul1 = psi1[(i + resz) % num];
    mul2 = psi2[(i + resz) % num];
    summ = make_float2((c1.x * mul1.x - c1.y * mul1.y + c2.x * mul2.x - c2.y * mul2.y), 
                      (c1.x * mul1.y + c1.y * mul1.x + c2.x * mul2.y + c2.y * mul2.x));
    if (summ.x > 0) {
        result = atan(summ.y, summ.x);
    }
    else if (summ.y > 0) {
        result = pi - atan(summ.y, -summ.x);
    }
    else {
        result = -pi + atan(summ.y, summ.x);
    }
    vy[i] = (float)result * hbar;
    
    mul1 = psi1[(i + resz * resy) % num];
    mul2 = psi2[(i + resz * resy) % num];
    summ = make_float2((c1.x * mul1.x - c1.y * mul1.y + c2.x * mul2.x - c2.y * mul2.y), 
                      (c1.x * mul1.y + c1.y * mul1.x + c2.x * mul2.y + c2.y * mul2.x));
    if (summ.x > 0) {
        result = atan(summ.y, summ.x);
    }
    else if (summ.y > 0) {
        result = pi - atan(summ.y, -summ.x);
    }
    else {
        result = -pi + atan(summ.y, summ.x);
    }
    vx[i] = (float)result * hbar;*/
}