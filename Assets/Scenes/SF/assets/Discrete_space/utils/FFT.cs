using System.Numerics;
using ManagedCuda;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;

namespace source.assets.Discrete_space.utils
{
    static class FFT
    {
        private static CudaFFTPlan3D _plan3DC, _plan3DR;
        
        public static void init(int nx, int ny, int nz)
        {
            _plan3DC = new CudaFFTPlan3D(nx, ny, nz, cufftType.C2C);
            _plan3DR = new CudaFFTPlan3D(nx, ny, nz, cufftType.R2C);
        }

        public static void fft_c(CudaDeviceVariable<cuFloatComplex> data, bool inverse)
        {
            if (!inverse)
            {
                _plan3DC.Exec(data.DevicePointer, TransformDirection.Forward);
            }
            else
            {
                _plan3DC.Exec(data.DevicePointer, TransformDirection.Inverse);
            }
        }
        
        public static CudaDeviceVariable<cuFloatComplex> fft_r(CudaDeviceVariable<float> data, bool inverse)
        {
            var result = new CudaDeviceVariable<cuFloatComplex>(ISF.properties.num);
            ISFKernels.copy.Run(result.DevicePointer, data.DevicePointer);

            if (!inverse)
            {
                _plan3DC.Exec(result.DevicePointer, TransformDirection.Forward);
            }
            else
            {
                _plan3DC.Exec(result.DevicePointer, TransformDirection.Inverse);
            }

            return result;
        }
    }
}
