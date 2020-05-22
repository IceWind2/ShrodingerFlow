using System;
using ManagedCuda;

namespace source.assets.Particles.utils
{
    public abstract class Handler
    {
        protected static CudaDeviceVariable<float> x, y, z;
        
        public static void set_particles(CudaDeviceVariable<float> xx, CudaDeviceVariable<float> yy, CudaDeviceVariable<float> zz)
        {
            x = xx;
            y = yy;
            z = zz;
        }

        public static void init() { }

        public static void update_particles() { }

    }
}
