extern "C" __global__ void
update_velocities(const float* x, const float* y, const float* z, 
                  const float* shiftX, const float* shiftY, const float* shiftZ, const float* fact, 
                  float*** vx, float*** vy, float*** vz, 
                  float* _ux, float* _uy, float* _uz, 
                  int* tor_size, int* tor_res, float* tor_d)
        {
            int i = blockIdx.x;

            float sh_x = shiftX[i] * fact[0];
            float tmp_x = (int)(x[i] + sh_x) % tor_size[0];
            int ix = (int)floorf(tmp_x / tor_d[0]);
            int ixp = ix % tor_res[0];
            float wx = tmp_x - (ix - 1) * tor_d[0];

            //INIT Y indices
            float sh_y = shiftY[i] * fact[0];
            float tmp_y = (int)(y[i] + sh_y) % tor_size[1];
            int iy = (int)floorf(tmp_y / tor_d[1]);
            int iyp = iy % tor_res[1];
            float wy = tmp_y - (iy - 1) * tor_d[1];

            //INIT Z indices
            float sh_z = shiftZ[i] * fact[0];
            float tmp_z = (int)(z[i] + sh_z) % tor_size[2];
            int iz = (int)floorf(tmp_z / tor_d[2]);
            int izp = iz % tor_res[2];
            float wz = tmp_z - (iz - 1) * tor_d[2];

            //Calculate Velocities
            _ux[i] = (1 - wz) * ((1 - wy) * vx[ix][iy][iz] + wy * vx[ix][iyp][iz]) +
                     wz * ((1 - wy) * vx[ix][iy][izp] + wy * vx[ix][iyp][izp]);
            _uy[i] = (1 - wz) * ((1 - wx) * vy[ix][iy][iz] + wx * vy[ixp][iy][iz]) +
                     wz * ((1 - wx) * vy[ix][iy][izp] + wx * vy[ixp][iy][izp]);
            _uz[i] = (1 - wy) * ((1 - wx) * vz[ix][iy][iz] + wx * vz[ixp][iy][iz]) +
                     wy * ((1 - wx) * vz[ix][iyp][iz] + wx * vz[ixp][iyp][iz]);
        }