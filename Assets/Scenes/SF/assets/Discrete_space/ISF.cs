using System;
using System.Linq;
using System.Numerics;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using source.assets.Discrete_space.utils;

namespace source.assets.Discrete_space
{
    public static class ISF
    {
        public static SpaceProperties properties;
        public static CudaDeviceVariable<cuFloatComplex> psi1, psi2;

        private static int[] _ix, _iy, _iz;                // 1D index array (Индексный массив)
        private static int[,,] _iix, _iiy, _iiz;        // iix::Array{Int, 3}; iiy::Array{Int, 3}; iiz::Array{Int, 3}    3D index array
        private static CudaDeviceVariable<cuFloatComplex> _mask, _fac; // Fourier coefficient for solving Schroedinger eq (Коэффициент Фурье)
        private static float t = 0;
        private static float[,,] _sx, _sy, _sz;
            
        public static void Init(int[] volSize, int[] volRes, float hbar, float dt) // vol_size::NTuple{ 3}, vol_res::NTuple{3}, hbar, dt)
        { 
            properties = new SpaceProperties(volSize, volRes, hbar, dt);
            ISFKernels.Init(properties);
            
            psi1 = new CudaDeviceVariable<cuFloatComplex>(properties.num);
            psi2 = new CudaDeviceVariable<cuFloatComplex>(properties.num);
            
            FFT.init(properties.resx, properties.resy, properties.resz);
            
            _ix = Enumerable.Range(0, properties.resx).ToArray();
            _iy = Enumerable.Range(0, properties.resy).ToArray();
            _iz = Enumerable.Range(0, properties.resz).ToArray();
            var ii = Matlab.ndgrid(_ix, _iy, _iz);
            _iix = ii.x;
            _iiy = ii.y;
            _iiz = ii.z;
            
            properties.px = new float[_iix.GetLength(0), _iix.GetLength(1), _iix.GetLength(2)];
            properties.py = new float[_iiy.GetLength(0), _iiy.GetLength(1), _iiy.GetLength(2)];
            properties.pz = new float[_iiz.GetLength(0), _iiz.GetLength(1), _iiz.GetLength(2)];
            for (int i = 0; i < properties.resx; i++)
            {
                for (int j = 0; j < properties.resy; j++)
                {
                    for (int k = 0; k < properties.resz; k++)
                    {
                        properties.px[i, j, k] = (_iix[i, j, k]) * properties.dx;
                        properties.py[i, j, k] = (_iiy[i, j, k]) * properties.dy;
                        properties.pz[i, j, k] = (_iiz[i, j, k]) * properties.dz;
                    }
                }
            }
            
            _sx = _iix.Select3D((e, i, j, k) => (float)Math.Sin((float)Math.PI * e / properties.resx) / properties.dx);
            _sy = _iiy.Select3D((e, i, j, k) => (float)Math.Sin((float)Math.PI * e / properties.resy) / properties.dy);
            _sz = _iiz.Select3D((e, i, j, k) => (float)Math.Sin((float)Math.PI * e / properties.resz) / properties.dz);
            
            cuFloatComplex[,,] tmpFac = _iix.Select3D((e, i, j, k) =>
            {
                return new cuFloatComplex((float)(-0.25 / (Math.Pow(_sx[i, j, k], 2) + Math.Pow(_sy[i, j, k], 2) + Math.Pow(_sz[i, j, k], 2))), 0);
            });
            tmpFac[0, 0, 0] = new cuFloatComplex(0, 0);
            _fac = new CudaDeviceVariable<cuFloatComplex>(properties.num);
            _fac.CopyToDevice(tmpFac);

            var tmpMask = new cuFloatComplex[properties.resx, properties.resy, properties.resz];
            build_schroedinger(tmpMask);
            _mask = new CudaDeviceVariable<cuFloatComplex>(properties.num);
            _mask.CopyToDevice(tmpMask);
        }
        
        public static void update_space()
        {
            t += properties.dt;
            
            schroedinger_flow();
            Normalize();
            PressureProject();
        }

        public static void update_velocities(Velocity vel)
        {
            velocity_oneForm(vel, properties.hbar);
            staggered_sharp(vel);
        }
        
        private static void build_schroedinger(cuFloatComplex[,,] tmpMask)
        {
            var fac = -4 * Math.Pow(Math.PI, 2) * properties.hbar;

            float kx, ky, kz;
            for (int i = 0; i < _iix.GetLength(0); i++)
            {
                for (int j = 0; j < _iix.GetLength(1); j++)
                {
                    for (int k = 0; k < _iix.GetLength(2); k++)
                    {
                        kx = (_iix[i, j, k] - (float)properties.resx / 2) / properties.sizex;
                        ky = (_iiy[i, j, k] - (float)properties.resy / 2) / properties.sizey;
                        kz = (_iiz[i, j, k] - (float)properties.resz / 2) / properties.sizez;

                        var lambda = fac * (Math.Pow(kx, 2) + Math.Pow(ky, 2) + Math.Pow(kz, 2));
                        var tmp = Complex.Exp(Complex.ImaginaryOne * lambda * properties.dt / 2f);
                        tmpMask[i, j, k] = new cuFloatComplex((float)tmp.Real, (float)tmp.Imaginary);
                    }
                }
            }
        }

        private static void schroedinger_flow()
        {
            FFT.fft_c(psi1, false);
            FFT.fft_c(psi2, false);

            ISFKernels.shift.Run(psi1.DevicePointer, properties.resx, properties.resy, properties.resz, properties.num);
            ISFKernels.shift.Run(psi2.DevicePointer, properties.resx, properties.resy, properties.resz, properties.num);

            ISFKernels.mul_each.Run(psi1.DevicePointer, _mask.DevicePointer);
            ISFKernels.mul_each.Run(psi2.DevicePointer, _mask.DevicePointer);

            ISFKernels.shift.Run(psi1.DevicePointer, properties.resx, properties.resy, properties.resz, properties.num);
            ISFKernels.shift.Run(psi2.DevicePointer, properties.resx, properties.resy, properties.resz, properties.num);
            
            FFT.fft_c(psi1, true);
            FFT.fft_c(psi2, true);

            ISFKernels.fft_norm.Run(psi1.DevicePointer, properties.num);
            ISFKernels.fft_norm.Run(psi2.DevicePointer, properties.num);
        }
        
        private static void staggered_sharp(Velocity vel)
        {
            ISFKernels.staggered.Run(vel.vx.DevicePointer, properties.dx);
            ISFKernels.staggered.Run(vel.vy.DevicePointer, properties.dy);
            ISFKernels.staggered.Run(vel.vz.DevicePointer, properties.dz);
        }
        
        private static void velocity_oneForm(Velocity v,  float hbar = 1)
        {
            ISFKernels.velocity_one.Run(psi1.DevicePointer, psi2.DevicePointer,
                              properties.resy, properties.resz, properties.num, hbar, (float) Math.PI,
                              v.vx.DevicePointer, v.vy.DevicePointer, v.vz.DevicePointer);
        }
        
        private static CudaDeviceVariable<float> Div(Velocity v)
        {
            var res = new CudaDeviceVariable<float>(properties.num);
            
            ISFKernels.div.Run(res.DevicePointer,
                                       v.vx.DevicePointer, v.vy.DevicePointer, v.vz.DevicePointer,
                                       properties.dx, properties.dy, properties.dz,
                                       properties.resx, properties.resy, properties.resz, properties.num);
            return res;
        }
        
        private static CudaDeviceVariable<cuFloatComplex> PoissonSolve(CudaDeviceVariable<float> f)
        {
            var res = FFT.fft_r(f, false);
        
            ISFKernels.mul_each.Run(res.DevicePointer, _fac.DevicePointer);

            FFT.fft_c(res, true);
            
            return res;
        }
        
        public static void PressureProject()
        {
            var v = new Velocity(properties.resx, properties.resy, properties.resz);
            velocity_oneForm(v);
            var div = Div(v);
            
            var q = PoissonSolve(div);

            ISFKernels.gauge.Run(psi1.DevicePointer, psi2.DevicePointer, q.DevicePointer, properties.num);

            v.vx.Dispose();
            v.vy.Dispose();
            v.vz.Dispose();
            div.Dispose();
            q.Dispose();
        }
        
        public static void Normalize()
        {
            ISFKernels.normalize.Run(psi1.DevicePointer, psi2.DevicePointer);
        }
        
        public static void add_circle(cuFloatComplex[,,] psi, float[] center, float[] normal, float r, float d)
        {
            float norm = (float)Math.Sqrt(Math.Pow(normal[0], 2) + Math.Pow(normal[1], 2) + Math.Pow(normal[2], 2));
            for (int i = 0; i < 3; i++)
            {
                normal[i] /= norm;
            }
            
            float alpha, rx, ry, rz, z;
            Complex tmp;
            for (int i = 0; i < properties.resx; i++)
            {
                for (int j = 0; j < properties.resy; j++)
                {
                    for (int k = 0; k < properties.resz; k++)
                    {
                        rx = properties.px[i, j, k] - center[0];
                        ry = properties.py[i, j, k] - center[1];
                        rz = properties.pz[i, j, k] - center[2];
                        alpha = 0;
                        z = rx * normal[0] + ry * normal[1] + rz * normal[2];
                        if (rx * rx + ry * ry + rz * rz - z * z < r * r)
                        {
                            if (z > 0 && z <= d / 2)
                            {
                                alpha = (float) -Math.PI * (2 * z / d - 1);
                            }

                            if (z <= 0 && z >= -d / 2)
                            {
                                alpha = (float) -Math.PI * (2 * z / d + 1);
                            }
                        }

                        tmp = new Complex(psi[i, j, k].real, psi[i, j, k].imag);
                        tmp *= Complex.Exp(Complex.ImaginaryOne * alpha);
                        psi[i, j, k] = new cuFloatComplex((float)tmp.Real, (float)tmp.Imaginary);
                    }
                }
            }
        }
    }
}