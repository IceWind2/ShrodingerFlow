using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;

namespace source.assets.Particles.utils
{
    class ParticleSet
    {
        public CudaDeviceVariable<float> x, y, z;
        public CudaDeviceVariable<float> vx, vy, vz;
        public int size;

        public ParticleSet(float[] xx, float[] yy, float[] zz, float[] vxx, float[] vyy, float[] vzz)
        {
            x = xx;
            y = yy;
            z = zz;
            vx = vxx;
            vy = vyy;
            vz = vzz;
            size = xx.Length;
        }
    }
}
