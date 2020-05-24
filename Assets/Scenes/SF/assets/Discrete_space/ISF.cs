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
            
            var tmp1 = new cuFloatComplex[properties.resx, properties.resy, properties.resz];
            var tmp2 = new cuFloatComplex[properties.resx, properties.resy, properties.resz];
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

                        tmp1[i, j, k] = new cuFloatComplex(1, 0);
                        tmp2[i, j, k] = new cuFloatComplex((float)0.01, 0);
                    }
                }
            }
            
            _sx = _iix.Select3D((e, i, j, k) => (float)Math.Sin((float)Math.PI / properties.resx) / properties.dx);
            _sy = _iiy.Select3D((e, i, j, k) => (float)Math.Sin((float)Math.PI / properties.resy) / properties.dy);
            _sz = _iiz.Select3D((e, i, j, k) => (float)Math.Sin((float)Math.PI / properties.resz) / properties.dz);
            
            cuFloatComplex[,,] tmpFac = _iix.Select3D((e, i, j, k) =>
            {
                return new cuFloatComplex((float)(-0.25 / (Math.Pow(_sx[i, j, k], 2) + Math.Pow(_sy[i, j, k], 2) + Math.Pow(_sz[i, j, k], 2))), 0);
            });
            tmpFac[0, 0, 0] = new cuFloatComplex(0, 0);
            _fac = new CudaDeviceVariable<cuFloatComplex>(properties.num);
            _fac.CopyToDevice(tmpFac);
            
            psi1.CopyToDevice(tmp1);
            psi2.CopyToDevice(tmp2);
            Normalize();
            
            var tmpMask = new cuFloatComplex[properties.resx, properties.resy, properties.resz];
            build_schroedinger(tmpMask);
            _mask = new CudaDeviceVariable<cuFloatComplex>(properties.num);
            _mask.CopyToDevice(tmpMask);
        }
        
        public static void update_space(Simulation s)
        {
            t += properties.dt;
            
            schroedinger_flow();
            Normalize();
            PressureProject();
            
            Constraint(s);
        }

        public static void update_velocities(Velocity vel)
        {
            velocity_oneForm(vel);
            staggered_sharp(vel);
        }
        
        private static void build_schroedinger(cuFloatComplex[,,] tmpMask)
        {
            var fac = -4 * Math.Pow(Math.PI, 2) * properties.hbar;
            
            float[,,] kx = new float[_iix.GetLength(0), _iix.GetLength(1), _iix.GetLength(2)];
            float[,,] ky = new float[_iiy.GetLength(0), _iiy.GetLength(1), _iiy.GetLength(2)];
            float[,,] kz = new float[_iiz.GetLength(0), _iiz.GetLength(1), _iiz.GetLength(2)];
            for (int i = 0; i < _iix.GetLength(0); i++)
            {
                for (int j = 0; j < _iix.GetLength(1); j++)
                {
                    for (int k = 0; k < _iix.GetLength(2); k++)
                    {
                        kx[i, j, k] = (_iix[i, j, k] - 1 - (float)properties.resx / 2) / properties.sizex;
                        kx[i, j, k] = (_iiy[i, j, k] - 1 - (float)properties.resy / 2) / properties.sizey;
                        kx[i, j, k] = (_iiz[i, j, k] - 1 - (float)properties.resz / 2) / properties.sizez;

                        var lambda = fac * (Math.Pow(kx[i, j, k], 2) + Math.Pow(ky[i, j, k], 2) + Math.Pow(kz[i, j, k], 2));
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
        }
        
        private static void staggered_sharp(Velocity vel)
        {
            ISFKernels.staggered.Run(vel.vx.DevicePointer, properties.dx);
            ISFKernels.staggered.Run(vel.vy.DevicePointer, properties.dy);
            ISFKernels.staggered.Run(vel.vz.DevicePointer, properties.dz);
        }
        
        private static void velocity_oneForm(Velocity v)
        {
            ISFKernels.velocity_one.Run(psi1.DevicePointer, psi2.DevicePointer,
                              properties.resy, properties.resz, properties.num, properties.hbar, (float) Math.PI,
                              v.vx.DevicePointer, v.vy.DevicePointer, v.vz.DevicePointer);
        }
        
        private static CudaDeviceVariable<float> Div(Velocity v)
        {
            var res = new CudaDeviceVariable<float>(properties.num);
            
            ISFKernels.div.Run(res.DevicePointer,
                                       v.vx.DevicePointer, v.vy.DevicePointer, v.vz.DevicePointer,
                                       properties.dx, properties.dy, properties.dz,
                                       properties.resx, properties.num);
            return res;
        }
        
        private static CudaDeviceVariable<cuFloatComplex> PoissonSolve(CudaDeviceVariable<float> f)
        {
            var res = FFT.fft_r(f, false);

            ISFKernels.mul_each.Run(res.DevicePointer, _fac.DevicePointer);

            FFT.fft_c(res, true);
            
            return res;
        }
        
        private static void PressureProject()
        {
            var v = new Velocity(properties.resx, properties.resy, properties.resz);
            velocity_oneForm(v);
            var div = Div(v);
            var q = PoissonSolve(div);

            ISFKernels.gauge.Run(psi1.DevicePointer, psi2.DevicePointer, q.DevicePointer);

      
            v.vx.Dispose();
            v.vy.Dispose();
            v.vz.Dispose();
            div.Dispose();
            q.Dispose();
        }
        
        private static void Normalize()
        {
            ISFKernels.normalize.Run(psi1.DevicePointer, psi2.DevicePointer);
        }

        public static void Constraint(Simulation s)
        {
            var tmp1 = new cuFloatComplex[properties.resx, properties.resy, properties.resz];
            var tmp2 = new cuFloatComplex[properties.resx, properties.resy, properties.resz];
            
            psi1.CopyToHost(tmp1);
            psi2.CopyToHost(tmp2);

            s.constraint(tmp1, tmp2, t);
            
            psi1.CopyToDevice(tmp1);
            psi2.CopyToDevice(tmp2);
            
            PressureProject();
        }
    }

}