using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using source.assets.CUDA_kernels;
using source.assets.Particles.utils;
using source.assets.Discrete_space;

namespace source.assets.Particles
{
    public static class Particles
    {
        public static CudaDeviceVariable<float> x, y, z;
        private static int _size, _maxCnt;

        public static void init(int max_particles, ISF torus)
        {
            KernelLoader.init();
            
            SizeT size = max_particles * sizeof(float);
            x = new CudaDeviceVariable<float>(size);
            y = new CudaDeviceVariable<float>(size);
            z = new CudaDeviceVariable<float>(size);

            _size = 0;
            _maxCnt = max_particles;
            
            Handler.set_particles(x, y, z);   

            UpdateHandler.init();
            
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
    }
}