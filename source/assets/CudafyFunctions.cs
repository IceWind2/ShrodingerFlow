using System;

using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace source.assets
{
    class CudafyFunctions
    {
        public static void load_functions(GPGPU gpu)
        {
            CudafyModule km = CudafyTranslator.Cudafy(eArchitecture.OpenCL12);
            gpu.LoadModule(km);
        }

        //Velocity functions
        [Cudafy]
        public static void staggered_velocity(GThread thread, float[] x, float[] y, float[] z, float[] shiftX, float[] shiftY, float[] shiftZ, float[] fact, float[,,] vx, float[,,] vy, float[,,] vz, float[] _ux, float[] _uy, float[] _uz, int[] tor_size, int[] tor_res, float[] tor_d)
        {
            int i = thread.blockIdx.x;

            //INIT X indices
            float sh_x = shiftX[i] * fact[0];
            float tmp_x = (int)(x[i] + sh_x) % tor_size[0];
            int ix = (int)Math.Floor(tmp_x / tor_d[0]);
            int ixp = ix % tor_res[0];
            float wx = tmp_x - (ix - 1) * tor_d[0];

            //INIT Y indices
            float sh_y = shiftY[i] * fact[0];
            float tmp_y = (int)(y[i] + sh_y) % tor_size[1];
            int iy = (int)Math.Floor(tmp_y / tor_d[1]);
            int iyp = iy % tor_res[1];
            float wy = tmp_y - (iy - 1) * tor_d[1];

            //INIT Z indices
            float sh_z = shiftZ[i] * fact[0];
            float tmp_z = (int)(z[i] + sh_z) % tor_size[2];
            int iz = (int)Math.Floor(tmp_z / tor_d[2]);
            int izp = iz % tor_res[2];
            float wz = tmp_z - (iz - 1) * tor_d[2];

            //Calculate Velocities
            _ux[i] = (1 - wz) * ((1 - wy) * vx[ix, iy, iz] + wy * vx[ix, iyp, iz]) +
                     wz * ((1 - wy) * vx[ix, iy, izp] + wy * vx[ix, iyp, izp]);
            _uy[i] = (1 - wz) * ((1 - wx) * vy[ix, iy, iz] + wx * vy[ixp, iy, iz]) +
                     wz * ((1 - wx) * vy[ix, iy, izp] + wx * vy[ixp, iy, izp]);
            _uz[i] = (1 - wy) * ((1 - wx) * vz[ix, iy, iz] + wx * vz[ixp, iy, iz]) +
                     wy * ((1 - wx) * vz[ix, iyp, iz] + wx * vz[ixp, iyp, iz]);
        }

        [Cudafy]
        public static void update(GThread thread, float[] x, float[] y, float[] z,
                                                            float[] k1x, float[] k1y, float[] k1z,
                                                            float[] k2x, float[] k2y, float[] k2z,
                                                            float[] k3x, float[] k3y, float[] k3z,
                                                            float[] k4x, float[] k4y, float[] k4z,
                                                            int[] size, float[] dt)
        {
            int i = thread.blockIdx.x;
            switch (thread.threadIdx.x)
            {
                case 0:
                    x[i] += (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) * dt[0] / 6;
                    break;
                case 1:
                    y[i] += (k1y[i] + 2 * k2y[i] + 2 * k3y[i] + k4y[i]) * dt[0] / 6;
                    break;
                case 2:
                    z[i] += (k1z[i] + 2 * k2z[i] + 2 * k3z[i] + k4z[i]) * dt[0] / 6;
                    break;
            }
        }


        //Update functions
        [Cudafy]
        public static void add(GThread thread, float[] x, float[] y, float[] z, float[] xx, float[] yy, float[] zz, int[] size)
        {
            int i = thread.blockIdx.x;

            x[i + size[0]] = xx[i];
            y[i + size[0]] = yy[i];
            z[i + size[0]] = zz[i];
        }
    }
}
