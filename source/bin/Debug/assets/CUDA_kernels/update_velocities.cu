extern "C" __global__ void
update_velocities(float* x, float* y, float* z, 
                  float* shiftX, float* shiftY, float* shiftZ, float fact, 
                  float* vx, float* vy, float* vz, 
                  float* _ux, float* _uy, float* _uz, 
                  int* tor_size, int* tor_res, float* tor_d)
        {
            int i = blockIdx.x;

            float sh_x = shiftX[i] * fact;
            float tmp_x = (x[i] + sh_x);
            int ix = floor(tmp_x / tor_d[0]);
            int ixp = (ix + 1) % tor_res[0];
            float wx = tmp_x - (ix) * tor_d[0];

            //INIT Y indices
            float sh_y = shiftY[i] * fact;
            float tmp_y = (y[i] + sh_y);
            int iy = floor(tmp_y / tor_d[1]);
            int iyp = (iy + 1) % tor_res[1];
            float wy = tmp_y - (iy) * tor_d[1];

            //INIT Z indices
            float sh_z = shiftZ[i] * fact;
            float tmp_z = (z[i] + sh_z);
            int iz = floor(tmp_z / tor_d[2]);
            int izp = (iz + 1) % tor_res[2];
            float wz = tmp_z - (iz) * tor_d[2];

            //Calculate Velocities
            _ux[i] = (1 - wz) * ((1 - wy) * vx[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz] + wy * vx[ix* tor_res[2] * tor_res[1] + iyp * tor_res[2] + iz]) +
                           wz * ((1 - wy) * vx[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + izp] + wy * vx[ix * tor_res[2] * tor_res[1] + iyp * tor_res[2] + izp]);
            _uy[i] = (1 - wz) * ((1 - wx) * vy[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz] + wx * vy[ixp * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz]) +
                           wz * ((1 - wx) * vy[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + izp] + wx * vy[ixp * tor_res[2] * tor_res[1] + iy * tor_res[2] + izp]);
            _uz[i] = (1 - wy) * ((1 - wx) * vz[ix * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz] + wx * vz[ixp * tor_res[2] * tor_res[1] + iy * tor_res[2] + iz]) +
                           wy * ((1 - wx) * vz[ix * tor_res[2] * tor_res[1] + iyp * tor_res[2] + iz] + wx * vz[ixp * tor_res[2] * tor_res[1] + iyp * tor_res[2] + iz]);
        }