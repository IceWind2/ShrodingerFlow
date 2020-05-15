using System;
using Cudafy;
using Cudafy.Host;

using source.assets.Particles.utils;
using source.assets.Discrete_space;

namespace source.assets.Particles
{
    public static class Particles
    {
        public static float[] x, y, z;
        private static int _size, _maxCnt;
        private static GPGPU _gpu;

        public static void init(int max_particles, ISF torus, GPGPU gpu)
        {
            _gpu = gpu;

            x = gpu.Allocate<float>(max_particles);
            y = gpu.Allocate<float>(max_particles);
            z = gpu.Allocate<float>(max_particles);

            _size = 0;
            _maxCnt = max_particles;

            Handler.set_gpu(gpu);
            Handler.set_particles(x, y, z);   

            VelocityHandler.init(torus, _maxCnt);
        }

        public static void add_particles(int cnt, float[] nozzle_cen, float[] nozzle_rad)
        {
            var tmp = UpdateHandler.create_random(cnt);

            float[] xx = new float[cnt];
            float[] yy = new float[cnt];
            float[] zz = new float[cnt];

            for (int i = 0; i < cnt; i++)
            {
                xx[i] = nozzle_cen[0];
                yy[i] = (float)(nozzle_cen[1] + 0.9 * nozzle_rad[0] * tmp.cos[i]);
                zz[i] = (float)(nozzle_cen[2] + 0.9 * nozzle_rad[0] * tmp.sin[i]);
            }

            UpdateHandler.update_particles(xx, yy, zz, cnt, _size);

            _size += cnt;
        }

        public static void calculate_movement(float[,,] vx, float[,,] vy, float[,,] vz)
        {
            VelocityHandler.update_particles(vx, vy, vz, _size);
        }

        /*public static void Keep(int[] vol_size)
        {
            //todo: cudafy
        }*/
    }
}