extern "C" __global__ void
update_velocities(const float* x, const float* y, const float* z, 
                  const float* shiftX, const float* shiftY, const float* shiftZ, const float fact, 
                  float* vx, float* vy, float* vz, 
                  float* _ux, float* _uy, float* _uz, 
                  int* tor_size, int* tor_res, float* tor_d)
        {
            int i = blockIdx.x;

            float sh_x = shiftX[i] * fact;
            float tmp_x = (int)(x[i] + sh_x) % tor_size[0];
            int ix = (int)(tmp_x / tor_d[0]);
            int ixp = ix % tor_res[0];
            float wx = tmp_x - (ix) * tor_d[0];

            //INIT Y indices
            float sh_y = shiftY[i] * fact;
            float tmp_y = (int)(y[i] + sh_y) % tor_size[1];
            int iy = (int)(tmp_y / tor_d[1]);
            int iyp = iy % tor_res[1];
            float wy = tmp_y - (iy) * tor_d[1];

            //INIT Z indices
            float sh_z = shiftZ[i] * fact;
            float tmp_z = (int)(z[i] + sh_z) % tor_size[2];
            int iz = (int)(tmp_z / tor_d[2]);
            int izp = iz % tor_res[2];
            float wz = tmp_z - (iz) * tor_d[2];

            //Calculate Velocities
            _ux[i] = (1 - wz) * ((1 - wy) * vx[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz] + wy * vx[ix* tor_res[2] * tor_res[1] + iyp * tor_res[2] + iz]) +
                     wz * ((1 - wy) * vx[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + izp] + wy * vx[ix * tor_res[2] * tor_res[1] + iyp * tor_res[2] + izp]);
            _uy[i] = (1 - wz) * ((1 - wx) * vy[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz] + wx * vy[ixp * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz]) +
                     wz * ((1 - wx) * vy[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + izp] + wx * vy[ixp * tor_res[2] * tor_res[1] + iy * tor_res[2] + izp]);
            _uz[i] = (1 - wy) * ((1 - wx) * vz[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz] + wx * vz[ixp * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz]) +
                     wy * ((1 - wx) * vz[ix * tor_res[2] * tor_res[1] + iyp * tor_res[2] + iz] + wx * vz[ixp * tor_res[2] * tor_res[1] + iyp * tor_res[2] + iz]);
        }