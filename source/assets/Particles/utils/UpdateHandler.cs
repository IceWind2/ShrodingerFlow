using System;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using source.assets.CUDA_kernels;

namespace source.assets.Particles.utils
{
    public class RandomCoordinates
    {
        public float[] cos { get; set; }
        public float[] sin { get; set; }
    }

    public class UpdateHandler : Handler
    {
        private static CudaKernel _gpu; 
        
        public new static void init()
        {
            _gpu = KernelLoader.load_kernel("add_particles");
        }
        
        public static RandomCoordinates create_random(int cnt)
        {
            Random rnd = new Random();
            float[] t_cos = new float[cnt];
            float[] t_sin = new float[cnt];
            for (int i = 0; i < cnt; i++)
            {
                t_cos[i] = (float)Math.Cos(rnd.NextDouble() * 2 * Math.PI);
                t_sin[i] = (float)Math.Sin(rnd.NextDouble() * 2 * Math.PI);
            }
            return new RandomCoordinates { cos = t_cos, sin = t_sin };
        }

        public static void update_particles(float[] xx, float[] yy, float[] zz, int cnt, int size)
        {
            CudaDeviceVariable<float> d_xx = xx;
            CudaDeviceVariable<float> d_yy = yy;
            CudaDeviceVariable<float> d_zz = zz;

            _gpu.BlockDimensions = new dim3(1, 1, 1);
            _gpu.GridDimensions = new dim3(cnt, 1, 1);
            _gpu.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer, 
                                      d_xx.DevicePointer, d_yy.DevicePointer, d_zz.DevicePointer, 
                                      size);
            
            d_xx.Dispose();
            d_yy.Dispose();
            d_zz.Dispose();
        }
    }
}
