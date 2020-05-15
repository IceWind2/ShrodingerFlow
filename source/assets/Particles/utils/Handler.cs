using System;
using Cudafy.Host;

namespace source.assets.Particles.utils
{
    public abstract class Handler
    {
        protected static GPGPU _gpu;
        protected static float[] x, y, z;

        public static void set_gpu(GPGPU gpu)
        {
            _gpu = gpu;
        }

        public static void set_particles(float[] xx, float[] yy, float[] zz)
        {
            x = xx;
            y = yy;
            z = zz;
        }

        public static void init() { }

        public static void update_particles() { }

    }
}
