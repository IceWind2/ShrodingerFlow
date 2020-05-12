using System;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace source.assets
{
    public static class Particles
    {
        public static float[] x, y, z;
        private static int[] torus_res, torus_size;
        private static float[] torus_d;
        private static int[] _size;
        private static GPGPU _gpu;

        static Particles()
        {
            x = null;
            y = null;
            z = null;
            _size = new int[1] { 0 };
        }


        public static void Init(int max_particles, ISF torus, GPGPU gpu)
        {
            _gpu = gpu;
            x = _gpu.Allocate<float>(max_particles);
            y = _gpu.Allocate<float>(max_particles);
            z = _gpu.Allocate<float>(max_particles);
            torus_d = _gpu.CopyToDevice(new float[4] { torus.dx, torus.dy, torus.dz, torus.dt });
            torus_res = _gpu.CopyToDevice(new int[3] { torus.resx, torus.resy, torus.resz });
            torus_size = _gpu.CopyToDevice(new int[3] { torus.sizex, torus.sizey, torus.sizez });
        }

        public static void addPoints(int n_particles, float[] nozzle_cen, float[] nozzle_rad)
        {
            Random rnd = new Random();
            float[] t_cos = new float[_size[0]];
            float[] t_sin = new float[_size[0]];
            for (int i = 0; i < _size[0]; i++)
            {
                t_cos[i] = (float)Math.Cos(rnd.NextDouble() * 2 * Math.PI);
                t_sin[i] = (float)Math.Sin(rnd.NextDouble() * 2 * Math.PI);
            }
            float[] d_rad = _gpu.CopyToDevice(nozzle_rad);
            float[] d_cen = _gpu.CopyToDevice(nozzle_cen);
            float[] d_cos = _gpu.CopyToDevice(t_cos);
            float[] d_sin = _gpu.CopyToDevice(t_sin);
            int[] d_size = _gpu.CopyToDevice(_size);
            _gpu.Launch(1, n_particles, "add", x, y, z, d_cos, d_sin, d_cen, d_rad, d_size);
            _size[0] += n_particles;
        }

        [Cudafy]
        public static void add(GThread thread, float[] x, float[] y, float[] z, float[] cos, float[] sin, float[] nozzle_cen, float[] nozzle_rad, int[] size)
        {
            int i = thread.threadIdx.x + size[0];
            x[i] = nozzle_cen[0];
            y[i] = (float)(nozzle_cen[1] + 0.9 * nozzle_rad[0] * cos[i]);
            z[i] = (float)(nozzle_cen[2] + 0.9 * nozzle_rad[0] * sin[i]);
        }

        public static void Keep(int[] vol_size)
        {
            //todo: cudafy
        }

        public static void StaggeredAdvect(float[,,] vx, float[,,] vy, float[,,] vz)
        {
            int[] d_size = _gpu.CopyToDevice(_size);
            float[] d_dt = _gpu.CopyToDevice(new float[1] { 0 });

            float[,,] d_vx = _gpu.CopyToDevice(vx);
            float[,,] d_vy = _gpu.CopyToDevice(vy);
            float[,,] d_vz = _gpu.CopyToDevice(vz);

            float[] d_k1x = _gpu.Allocate<float>(_size[0]);
            float[] d_k1y = _gpu.Allocate<float>(_size[0]);
            float[] d_k1z = _gpu.Allocate<float>(_size[0]);
            _gpu.Launch(1, _size[0], "StaggeredVelocity", x, y, z, null, null, null, d_dt, d_vx, d_vy, d_vz, d_k1x, d_k1y, d_k1z, torus_size, torus_res, torus_d);

            float[] d_k2x = _gpu.Allocate<float>(_size[0]);
            float[] d_k2y = _gpu.Allocate<float>(_size[0]);
            float[] d_k2z = _gpu.Allocate<float>(_size[0]);
            d_dt[0] = torus_d[3] * (float)0.5;
            _gpu.Launch(1, _size[0], "StaggeredVelocity", x, y, z, d_k1x, d_k1y, d_k1z, d_dt, d_vx, d_vy, d_vz, d_k2x, d_k2y, d_k2z, torus_size, torus_res, torus_d);

            float[] d_k3x = _gpu.Allocate<float>(_size[0]);
            float[] d_k3y = _gpu.Allocate<float>(_size[0]);
            float[] d_k3z = _gpu.Allocate<float>(_size[0]);
            _gpu.Launch(1, _size[0], "StaggeredVelocity", x, y, z, d_k2x, d_k2y, d_k2z, d_dt, d_vx, d_vy, d_vz, d_k3x, d_k3y, d_k3z, torus_size, torus_res, torus_d);

            float[] d_k4x = _gpu.Allocate<float>(_size[0]);
            float[] d_k4y = _gpu.Allocate<float>(_size[0]);
            float[] d_k4z = _gpu.Allocate<float>(_size[0]);
            d_dt[0] = torus_d[3];
            _gpu.Launch(1, _size[0], "StaggeredVelocity", x, y, z, d_k3x, d_k3y, d_k3z, d_dt, d_vx, d_vy, d_vz, d_k4x, d_k4y, d_k4z, torus_size, torus_res, torus_d);

            _gpu.Launch(1, _size[0] * 3, "UpdateParticles", x, y, z, d_k1x, d_k1y, d_k1z, d_k2x, d_k2y, d_k2z, d_k3x, d_k3y, d_k3z,
                d_k4x, d_k4y, d_k4z);
        }

        [Cudafy]
        private static void StaggeredVelocity(GThread thread, float[] x, float[] y, float[] z, float[] shiftX, float[] shiftY, float[] shiftZ, float[] fact, float[,,] vx, float[,,] vy, float[,,] vz, float[] _ux, float[] _uy, float[] _uz, int[] tor_size, int[] tor_res, float[] tor_d)
        {
            int i = thread.threadIdx.x;

            //INIT X indices
            float sh_x = shiftX == null ? 0 : shiftX[i] * fact[0];
            float tmp_x = (int)(x[i] + sh_x) % tor_size[0];
            int ix = (int)Math.Floor(tmp_x / tor_d[0]);
            int ixp = ix % tor_res[0];
            float wx = tmp_x - (ix - 1) * tor_d[0];

            //INIT Y indices
            float sh_y = shiftY == null ? 0 : shiftY[i] * fact[0];
            float tmp_y = (int)(y[i] + sh_y) % tor_size[1];
            int iy = (int)Math.Floor(tmp_y / tor_d[1]);
            int iyp = iy % tor_res[1];
            float wy = tmp_y - (iy - 1) * tor_d[1];

            //INIT Z indices
            float sh_z = shiftZ == null ? 0 : shiftZ[i] * fact[0];
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
        private static void UpdateParticles(GThread thread, float[] x, float[] y, float[] z,
                                                            float[] k1x, float[] k1y, float[] k1z,
                                                            float[] k2x, float[] k2y, float[] k2z,
                                                            float[] k3x, float[] k3y, float[] k3z,
                                                            float[] k4x, float[] k4y, float[] k4z,
                                                            int[] size, float[] dt)
        {
            int i = thread.threadIdx.x;
            switch (i / size[0])
            {
                case 0:
                    x[i] += (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) * dt[0] / 6;
                    break;
                case 1:
                    i -= size[0];
                    x[i] += (k1y[i] + 2 * k2y[i] + 2 * k3y[i] + k4y[i]) * dt[0] / 6;
                    break;
                case 2:
                    i -= 2 * size[0];
                    z[i] += (k1z[i] + 2 * k2z[i] + 2 * k3z[i] + k4z[i]) * dt[0] / 6;
                    break;
            }
        }
    }

}