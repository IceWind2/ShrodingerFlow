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
            x = new CudaDeviceVariable<float>(max_particles);
            y = new CudaDeviceVariable<float>(max_particles);
            z = new CudaDeviceVariable<float>(max_particles);

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

        public static void calculate_movement(Velocity vel)
        {
            VelocityHandler.update_particles(vel.vx, vel.vy, vel.vz, _size);
        }
    }
}