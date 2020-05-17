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

        public static void init(ISF torus, int maxCnt)
        {
            _gpuVelocity = KernelLoader.load_kernel("update_velocities");
            _gpuUpdate = KernelLoader.load_kernel("update_particles");
            
            _dt = torus.dt;

            torus_d = new float[3] {torus.dx, torus.dy, torus.dz};
            torus_res = new int[3] {torus.resx, torus.resy, torus.resz};
            torus_size = new int[3] { torus.sizex, torus.sizey, torus.sizez };

            d_k1x = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k1y = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k1z = new CudaDeviceVariable<float>(maxCnt * sizeof(float));

            d_k2x = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k2y = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k2z = new CudaDeviceVariable<float>(maxCnt * sizeof(float));

            d_k3x = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k3y = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k3z = new CudaDeviceVariable<float>(maxCnt * sizeof(float));

            d_k4x = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k4y = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
            d_k4z = new CudaDeviceVariable<float>(maxCnt * sizeof(float));
        }

        public static void update_particles(float[,,] vx, float[,,] vy, float[,,] vz, int cnt)
        {
            int num = vx.GetLength(0) * vx.GetLength(1) * vx.GetLength(2);
            
            float d_dt = 0;

            CudaDeviceVariable<float> d_vx = new CudaDeviceVariable<float>(sizeof(float) * num);
            d_vx.CopyToDevice(vx);
            CudaDeviceVariable<float> d_vy = new CudaDeviceVariable<float>(sizeof(float) * num);
            d_vx.CopyToDevice(vx);
            CudaDeviceVariable<float> d_vz = new CudaDeviceVariable<float>(sizeof(float) * num);
            d_vx.CopyToDevice(vx);

            _gpuVelocity.BlockDimensions = new dim3(1, 1, 1);
            _gpuVelocity.GridDimensions = new dim3(cnt, 1, 1);
            
            _gpuVelocity.Run(x, y, z, d_k1x, d_k1y, d_k1z, d_dt, d_vx, d_vy, d_vz, d_k1x, d_k1y, d_k1z, torus_size, torus_res, torus_d);

            d_dt = _dt * (float)0.5;
            _gpuVelocity.Run(x, y, z, d_k1x, d_k1y, d_k1z, d_dt, d_vx, d_vy, d_vz, d_k2x, d_k2y, d_k2z, torus_size, torus_res, torus_d);

            _gpuVelocity.Run(x, y, z, d_k2x, d_k2y, d_k2z, d_dt, d_vx, d_vy, d_vz, d_k3x, d_k3y, d_k3z, torus_size, torus_res, torus_d);
            
            d_dt = _dt;
            _gpuVelocity.Run(x, y, z, d_k3x, d_k3y, d_k3z, d_dt, d_vx, d_vy, d_vz, d_k4x, d_k4y, d_k4z, torus_size, torus_res, torus_d);

            _gpuUpdate.BlockDimensions = new dim3(3, 1, 1);
            _gpuUpdate.GridDimensions = new dim3(cnt, 1, 1);
            
            _gpuUpdate.Run(x, y, z, d_k1x, d_k1y, d_k1z, d_k2x, d_k2y, d_k2z, d_k3x, d_k3y, d_k3z,
                d_k4x, d_k4y, d_k4z, d_dt);
            
            d_vx.Dispose();
            d_vy.Dispose();
            d_vz.Dispose();
        }   
    }
}
