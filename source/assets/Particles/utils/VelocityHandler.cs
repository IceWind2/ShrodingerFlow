﻿using System;
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
        private static int num;
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
            num = torus_res[0] * torus_res[1] * torus_res[2];
            
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
            float d_dt = 0;

            CudaDeviceVariable<float> d_vx = flatten(vx);
            CudaDeviceVariable<float> d_vy = flatten(vy);
            CudaDeviceVariable<float> d_vz = flatten(vz);

            _gpuVelocity.BlockDimensions = new dim3(1, 1, 1);
            _gpuVelocity.GridDimensions = new dim3(cnt, 1, 1);
            
            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer, 
                                              d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer, 
                                              d_dt, 
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer, 
                                              d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer, 
                                              torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);

            d_dt = _dt * (float)0.5;
            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer, 
                                              d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer, 
                                              d_dt, 
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer, 
                                              d_k2x.DevicePointer, d_k2y.DevicePointer, d_k2z.DevicePointer, 
                                              torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);
            
            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer, 
                                              d_k2x.DevicePointer, d_k2y.DevicePointer, d_k2z.DevicePointer, 
                                              d_dt, 
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer, 
                                              d_k3x.DevicePointer, d_k3y.DevicePointer, d_k3z.DevicePointer, 
                                              torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);
            
            d_dt = _dt;
            _gpuVelocity.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer, 
                                              d_k3x.DevicePointer, d_k3y.DevicePointer, d_k3z.DevicePointer, 
                                              d_dt, 
                                              d_vx.DevicePointer, d_vy.DevicePointer, d_vz.DevicePointer, 
                                              d_k4x.DevicePointer, d_k4y.DevicePointer, d_k4z.DevicePointer,
                                              torus_size.DevicePointer, torus_res.DevicePointer, torus_d.DevicePointer);
            
            _gpuUpdate.BlockDimensions = new dim3(3, 1, 1);
            _gpuUpdate.GridDimensions = new dim3(cnt, 1, 1);
            
            _gpuUpdate.Run(x.DevicePointer, y.DevicePointer, z.DevicePointer, 
                                            d_k1x.DevicePointer, d_k1y.DevicePointer, d_k1z.DevicePointer, 
                                            d_k2x.DevicePointer, d_k2y.DevicePointer, d_k2z.DevicePointer, 
                                            d_k3x.DevicePointer, d_k3y.DevicePointer, d_k3z.DevicePointer, 
                                            d_k4x.DevicePointer, d_k4y.DevicePointer, d_k4z.DevicePointer, 
                                            d_dt);
            
            d_vx.Dispose();
            d_vy.Dispose();
            d_vz.Dispose();
        }

        private static float[] flatten(float[,,] arr)
        {
            var res = new float[arr.GetLength(0) * arr.GetLength(1) * arr.GetLength(2)];
            
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    for (int k = 0; k < arr.GetLength(2); k++)
                    {
                        res[i * arr.GetLength(1) * arr.GetLength(2) + j * arr.GetLength(2) + k] = arr[i, j, k];
                    }
                }
            }

            return res;
        }
    }
}