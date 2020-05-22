using ManagedCuda;
using ManagedCuda.BasicTypes;
using source.assets.Particles.utils;

namespace source.assets.Particles
{
    public static class Particles
    {
        public static CudaDeviceVariable<float> x, y, z;
        private static int _size, _maxCnt;

        public static void init(int max_particles)
        {
            SizeT size = max_particles * sizeof(float);
            x = new CudaDeviceVariable<float>(size);
            y = new CudaDeviceVariable<float>(size);
            z = new CudaDeviceVariable<float>(size);

            _size = 0;
            _maxCnt = max_particles;
            
            Handler.set_particles(x, y, z);   

            UpdateHandler.init();
            
            VelocityHandler.init(_maxCnt);
        }

        public static void add_particles(float[] xx, float[] yy, float[] zz, int cnt)
        {
            UpdateHandler.update_particles(xx, yy, zz, cnt, _size);

            _size += cnt;
        }

        public static void calculate_movement(CudaDeviceVariable<float> vx, CudaDeviceVariable<float> vy, CudaDeviceVariable<float> vz)
        {
            VelocityHandler.update_particles(vx, vy, vz, _size);
        }
    }
}