using System;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using source.assets.CUDA_kernels;
using source.assets.Discrete_space;

namespace source.assets.Particles.utils
{
    public class VelocityHandler : Handler
    {
        private static CudaDeviceVariable<int> torus_res, torus_size;
        private static CudaDeviceVariable<float> torus_d;
        private static float _dt;
        private static CudaDeviceVariable<float> d_k1x, d_k1y, d_k1z, d_k2x, d_k2y, d_k2z, d_k3x, d_k3y, d_k3z, d_k4x, d_k4y, d_k4z;
        private static CudaKernel _gpuVelocity, _gpuUpdate; 

        public static void init(int maxCnt)
        {
            _gpuVelocity = KernelLoader.load_kernel("update_velocities");
            _gpuUpdate = KernelLoader.load_kernel("update_particles");
            
            _dt = ISF.properties.dt;
 
            torus_d = new float[3] {ISF.properties.dx, ISF.properties.dy, ISF.properties.dz};
            torus_res = new int[3] {ISF.properties.resx, ISF.properties.resy, ISF.properties.resz};
	    torus_size = new int[3] {ISF.properties.sizex, ISF.properties.sizey, ISF.properties.sizez};

            d_k1x = new CudaDeviceVariable<float>(maxCnt);
            d_k1y = new CudaDeviceVariable<float>(maxCnt);
            d_k1z = new CudaDeviceVariable<float>(maxCnt);

            d_k2x = new CudaDeviceVariable<float>(maxCnt);
            d_k2y = new CudaDeviceVariable<float>(maxCnt);
            d_k2z = new CudaDeviceVariable<float>(maxCnt);

            d_k3x = new CudaDeviceVariable<float>(maxCnt);
            d_k3y = new CudaDeviceVariable<float>(maxCnt);
            d_k3z = new CudaDeviceVariable<float>(maxCnt);

            d_k4x = new CudaDeviceVariable<float>(maxCnt);
            d_k4y = new CudaDeviceVariable<float>(maxCnt);
            d_k4z = new CudaDeviceVariable<float>(maxCnt);
        }

        public static void update_particles(CudaDeviceVariable<float> d_vx, CudaDeviceVariable<float> d_vy, CudaDeviceVariable<float> d_vz, int cnt)
        {
            float d_dt = 0;

            int thrd = (int)Math.Pow(2, Math.Ceiling(Math.Log(Math.Sqrt(Math.Sqrt(cnt))) / Math.Log(2)));

            _gpuVelocity.BlockDimensions = new dim3(thrd, thrd, 1);
            _gpuVelocity.GridDimensions = new dim3((int)Math.Ceiling(Math.Sqrt(cnt / thrd / thrd)), (int)Math.Ceiling(Math.Sqrt(cnt / thrd / thrd)), 1);

            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer,
                                              d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer,
                                              d_dt,
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer,
                                              d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer,
                                              cnt, torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);

            d_dt = _dt * (float)0.5;
            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer,
                                              d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer,
                                              d_dt,
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer,
                                              d_k2x.DevicePointer, d_k2y.DevicePointer, d_k2z.DevicePointer,
                                              cnt, torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);

            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer,
                                              d_k2x.DevicePointer, d_k2y.DevicePointer, d_k2z.DevicePointer,
                                              d_dt,
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer,
                                              d_k3x.DevicePointer, d_k3y.DevicePointer, d_k3z.DevicePointer,
                                              cnt, torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);

            d_dt = _dt;
            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer,
                                              d_k3x.DevicePointer, d_k3y.DevicePointer, d_k3z.DevicePointer,
                                              d_dt,
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer,
                                              d_k4x.DevicePointer, d_k4y.DevicePointer, d_k4z.DevicePointer,
                                              cnt, torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);


            _gpuUpdate.BlockDimensions = new dim3(thrd, thrd, 1);
            _gpuUpdate.GridDimensions = new dim3((int)Math.Ceiling(Math.Sqrt(cnt / thrd / thrd)) * 3, (int)Math.Ceiling(Math.Sqrt(cnt / thrd / thrd)), 1);

            _gpuUpdate.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer,
                                            d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer,
                                            d_k2x.DevicePointer, d_k2y.DevicePointer, d_k2z.DevicePointer,
                                            d_k3x.DevicePointer, d_k3y.DevicePointer, d_k3z.DevicePointer,
                                            d_k4x.DevicePointer, d_k4y.DevicePointer, d_k4z.DevicePointer,
                                            cnt, d_dt);
        }
    }
}
