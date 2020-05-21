using System;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using source.assets.CUDA_kernels;

namespace source.assets.Particles.utils
{
    public class UpdateHandler : Handler
    {
        private static CudaKernel _gpu; 
        
        public new static void init()
        {
            _gpu = KernelLoader.load_kernel("add_particles");
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
