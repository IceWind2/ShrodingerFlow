using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using source.assets.Discrete_space.utils;

namespace source.assets.Discrete_space
{
    public class ISF
    {
        //private int currentResolution;
        //private ParticleSystem.Particle[] points;

        public float[,,] px, py, pz;         // px::Array{float64,3}; py::Array{float64,3}; pz::Array{float64,3}  coordinates of grid points
        public int[] ix, iy, iz;                // 1D index array (Индексный массив)
        public int[,,] iix, iiy, iiz;        // iix::Array{Int, 3}; iiy::Array{Int, 3}; iiz::Array{Int, 3}    3D index array
        public float dx, dy, dz;                    // edge length
        public int sizex, sizey, sizez;             // size of grid
        public int resx, resy, resz;             // number of grid points in each dimension (Количество точек сетки в каждом измерении)

        public float hbar;                  // reduced Planck constant (Понижение постоянной Планка)
        public float dt;                    // time step
        public cuFloatComplex[,,] tmp_mask; // tmp_mask::Array{ Complex{ float64},3}  // Fourier coefficient for solving Schroedinger eq (Коэффициент Фурье)
        //public CudaDeviceVariable<cuFloatComplex> mask;
        
        public ISF(int[] vol_size, int[] vol_res, float _hbar, float _dt) // vol_size::NTuple{ 3}, vol_res::NTuple{3}, hbar, dt)
        {
            //obj = new()

            sizex = vol_size[0];
            sizey = vol_size[1];
            sizez = vol_size[2]; //obj.sizex, obj.sizey, obj.sizez = vol_size

            resx = vol_res[0];
            resy = vol_res[1];
            resz = vol_res[2];

            dx = sizex / (float)resx;
            dy = sizey / (float)resy;
            dz = sizez / (float)resz;

            ix = Enumerable.Range(1, resx).ToArray();
            iy = Enumerable.Range(1, resy).ToArray();
            iz = Enumerable.Range(1, resz).ToArray();
            var ii = Matlab.ndgrid(ix, iy, iz);
            iix = ii.x;
            iiy = ii.y;
            iiz = ii.z;

            px = new float[iix.GetLength(0), iix.GetLength(1), iix.GetLength(2)];
            py = new float[iiy.GetLength(0), iiy.GetLength(1), iiy.GetLength(2)];
            pz = new float[iiz.GetLength(0), iiz.GetLength(1), iiz.GetLength(2)];
            for (int i = 0; i < iix.GetLength(0); i++)
            {
                for (int j = 0; j < iix.GetLength(1); j++)
                {
                    for (int k = 0; k < iix.GetLength(2); k++)
                    {
                        px[i, j, k] = (iix[i, j, k] - 1) * dx;
                        py[i, j, k] = (iiy[i, j, k] - 1) * dy;
                        pz[i, j, k] = (iiz[i, j, k] - 1) * dz;
                    }
                }
            }

            hbar = _hbar;
            dt = _dt;
            tmp_mask = new cuFloatComplex[resx, resy, resz];
            build_schroedinger();
        }

        public void build_schroedinger() //private void CreatePoints()
        {
            var nx = this.resx;
            var ny = this.resy;
            var nz = this.resz;
            var fac = -4 * Math.Pow(Math.PI, 2) * this.hbar;

            float[,,] kx = new float[iix.GetLength(0), iix.GetLength(1), iix.GetLength(2)];
            float[,,] ky = new float[iiy.GetLength(0), iiy.GetLength(1), iiy.GetLength(2)];
            float[,,] kz = new float[iiz.GetLength(0), iiz.GetLength(1), iiz.GetLength(2)];
            for (int i = 0; i < iix.GetLength(0); i++)
            {
                for (int j = 0; j < iix.GetLength(1); j++)
                {
                    for (int k = 0; k < iix.GetLength(2); k++)
                    {
                        kx[i, j, k] = (iix[i, j, k] - 1 - nx / 2) / sizex;
                        kx[i, j, k] = (iiy[i, j, k] - 1 - ny / 2) / sizey;
                        kx[i, j, k] = (iiz[i, j, k] - 1 - nz / 2) / sizez;

                        var lambda = fac * (Math.Pow(kx[i, j, k], 2) + Math.Pow(ky[i, j, k], 2) + Math.Pow(kz[i, j, k], 2));
                        var tmp = Complex.Exp(Complex.ImaginaryOne * lambda * dt / 2f);
                        tmp_mask[i, j, k] = new cuFloatComplex((float)tmp.Real, (float)tmp.Imaginary);
                        
                    }
                }
            }
        }
        private const int minLength = 2;
        private const int maxLength = 16384;

        public void schroedinger_flow(Complex[,,] psi1, Complex[,,] psi2)
        {
            int[] dim = new int[3] { psi1.GetLength(0), psi1.GetLength(1), psi1.GetLength(2) };

            /*var tmp1 = _gpu.CopyToDevice(psi1);
            var tmp2 = _gpu.CopyToDevice(psi2);

            FFT.fft_c(tmp1, tmp1, dim, false);
            FFT.fft_c(tmp2, tmp2, dim, false);

            FFT.shift(tmp1, dim);
            FFT.shift(tmp2, dim);

            FFT.mul_each(tmp1, mask, dim);
            FFT.mul_each(tmp2, mask, dim);

            FFT.shift(tmp1, dim);
            FFT.shift(tmp2, dim);

            FFT.fft_c(tmp1, tmp1, dim, true);
            FFT.fft_c(tmp2, tmp2, dim, true);*/
        }

        // For a 1-form v compute the corresponding vector field `v^sharp` as a staggered vector field living on edges
        // Для 1-формы v вычислить соответствующее векторное поле `v ^ sharp` как шахматное векторное поле, живущее на краях
        public Velocity staggered_sharp(float[,,] _vx, float[,,] _vy, float[,,] _vz)
        {
            float[,,] ux = new float[_vx.GetLength(0), _vx.GetLength(1), _vx.GetLength(2)];
            float[,,] uy = new float[_vx.GetLength(0), _vx.GetLength(1), _vx.GetLength(2)]; ;
            float[,,] uz = new float[_vx.GetLength(0), _vx.GetLength(1), _vx.GetLength(2)]; ;

            for (int i = 0; i < _vx.GetLength(0); i++)
            {
                for (int j = 0; j < _vx.GetLength(1); j++)
                {
                    for (int k = 0; k < _vx.GetLength(2); k++)
                    {
                        ux[i, j, k] = _vx[i, j, k] / dx;
                        uy[i, j, k] = _vy[i, j, k] / dy;
                        uz[i, j, k] = _vz[i, j, k] / dz;
                    }
                }
            }

            return new Velocity { vx = ux, vy = uy, vz = uz };
        }

        public class Velocity
        {
            public float[,,] vx;
            public float[,,] vy;
            public float[,,] vz;
        }
        // Скорость одной формы
        public Velocity velocity_oneForm(Complex[,,] psi1, Complex[,,] psi2, float hbar = 1.0f)
        {
            //ixp = mod.(ix, resx) + 1;
            var ixp = ix.Select(i => i % resx).ToList();
            var iyp = iy.Select(i => i % resy).ToList();
            var izp = iz.Select(i => i % resz).ToList();

            //todo: check if reorder of x and z is needed?
            //fassuming psi are a square
            float[,,] vx = new float[psi1.GetLength(0), psi1.GetLength(1), psi1.GetLength(2)];
            //fassuming psi are a square
            float[,,] vy = new float[psi1.GetLength(0), psi1.GetLength(1), psi1.GetLength(2)];

            //fassuming psi are a square
            float[,,] vz = new float[psi1.GetLength(0), psi1.GetLength(1), psi1.GetLength(2)];

            for (int index1 = 0; index1 < psi1.GetLength(0); ++index1)
            {
                for (int index2 = 0; index2 < psi1.GetLength(1); ++index2)
                {
                    for (int index3 = 0; index3 < psi1.GetLength(2); ++index3)
                    {
                        var c1 = Complex.Conjugate(psi1[index1, index2, index3]);
                        var c2 = Complex.Conjugate(psi2[index1, index2, index3]);
                        var mul1 = psi1[index1, index2, izp[index3]];
                        var mul2 = psi1[index1, index2, izp[index3]];
                        var summ = (c1 * mul1 + c2 * mul2);
                        var result = summ.Phase;
                        vz[index1, index2, index3] = (float)result * hbar;
                    }
                }
            }


            for (int index1 = 0; index1 < psi1.GetLength(0); ++index1)
            {
                for (int index2 = 0; index2 < psi1.GetLength(1); ++index2)
                {
                    for (int index3 = 0; index3 < psi1.GetLength(2); ++index3)
                    {
                        var c1 = Complex.Conjugate(psi1[index1, index2, index3]);
                        var c2 = Complex.Conjugate(psi2[index1, index2, index3]);
                        var mul1 = psi1[index1, iyp[index2], index3];
                        var mul2 = psi1[index1, iyp[index2], index3];
                        var summ = (c1 * mul1 + c2 * mul2);
                        var result = summ.Phase;
                        vy[index1, index2, index3] = (float)result * hbar;
                    }
                }
            }

            for (int index1 = 0; index1 < psi1.GetLength(0); ++index1)
            {
                for (int index2 = 0; index2 < psi1.GetLength(1); ++index2)
                {
                    for (int index3 = 0; index3 < psi1.GetLength(2); ++index3)
                    {
                        var c1 = Complex.Conjugate(psi1[index1, index2, index3]);
                        var c2 = Complex.Conjugate(psi2[index1, index2, index3]);
                        var mul1 = psi1[ixp[index1], index2, index3];
                        var mul2 = psi1[ixp[index1], index2, index3];
                        var summ = (c1 * mul1 + c2 * mul2);
                        var result = summ.Phase;
                        vx[index1, index2, index3] = (float)result * hbar;
                    }
                }
            }

            return new Velocity { vx = vx, vy = vy, vz = vz };
        }

        public float[,,] Div(Velocity v)
        {
            var ixm = ix.Select(i => (i - 2 + resx) % resx).ToList();
            var iym = iy.Select(i => (i - 2 + resy) % resy).ToList();
            var izm = iz.Select(i => (i - 2 + resz) % resz).ToList();

            float[,,] f = new float[v.vx.GetLength(0), v.vx.GetLength(1), v.vx.GetLength(2)];
            for (int index1 = 0; index1 < v.vx.GetLength(0); ++index1)
            {
                for (int index2 = 0; index2 < v.vx.GetLength(1); ++index2)
                {
                    for (int index3 = 0; index3 < v.vx.GetLength(2); ++index3)
                    {
                        f[index1, index2, index3] = (v.vx[index1, index2, index3] - v.vx[ixm[index1], index2, index3]) / (float)Math.Pow(dx, 2);
                        f[index1, index2, index3] += (v.vy[index1, index2, index3] - v.vy[index1, iym[index2], index3]) / (float)Math.Pow(dy, 2);
                        f[index1, index2, index3] += (v.vz[index1, index2, index3] - v.vz[index1, index2, izm[index3]]) / (float)Math.Pow(dz, 2);
                    }
                }
            }

            return f;
        }

        // Распределение Пуассона
        public Complex[,,] PoissonSolve(float[,,] f)
        {
            int[] dim = new int[3] { f.GetLength(0), f.GetLength(1), f.GetLength(2) };

            Complex[,,] fc = new Complex[dim[0], dim[1], dim[2]];

            for (int i = 0; i < dim[0]; i++)
            {
                for (int j = 0; j < dim[1]; j++)
                {
                    for (int k = 0; k < dim[2]; k++)
                    {
                        fc[i, j, k] = 1;
                    }
                }
            }

            return fc;

            /*var d_f = _gpu.CopyToDevice(f);
            var d_fc = _gpu.CopyToDevice(fc);

            FFT.fft_r(d_f, d_fc, dim, false);


            var sx = iix.Select3D((e, i, j, k) => Math.Sin(Math.PI * (e - 1) / resx) / dx);
            var sy = iiy.Select3D((e, i, j, k) => Math.Sin(Math.PI * (e - 1) / resy) / dy);
            var sz = iiz.Select3D((e, i, j, k) => Math.Sin(Math.PI * (e - 1) / resz) / dz);

            Complex[,,] fac = iix.Select3D((e, i, j, k) =>
            {
                return (Complex)(-0.25 / (Math.Pow(sx[i, j, k], 2) + Math.Pow(sy[i, j, k], 2) + Math.Pow(sz[i, j, k], 2)));
            });
            fac[0, 0, 0] = 0;
            Complex[,,] d_fac = _gpu.CopyToDevice(fac);

            FFT.mul_each(d_fc, d_fac, dim);

            FFT.fft_c(d_fc, d_fc, dim, true);

            _gpu.CopyFromDevice(fc, d_fc);
            return fc;*/
        }

        public class Psi
        {
            public Complex[][][] psi1;
            public Complex[][][] psi2;
        }
        // Калибровочное преобразование
        public void GaugeTransform(Complex[,,] psi1, Complex[,,] psi2, Complex[,,] q)
        {
            var result = new Psi();
            var eiq = q.Select3D((e, i, j, k) => Complex.Exp(Complex.ImaginaryOne * e));
            for (int i = 0; i < psi1.GetLength(0); i++)
            {
                for (int j = 0; j < psi1.GetLength(1); j++)
                {
                    for (int k = 0; k < psi1.GetLength(2); k++)
                    {
                        psi1[i, j, k] *= eiq[i, j, k];
                        psi2[i, j, k] *= eiq[i, j, k];
                    }
                }
            }
        }

        // Давление
        public void PressureProject(Complex[,,] psi1, Complex[,,] psi2)
        {
            var v = velocity_oneForm(psi1, psi2);
            var div = Div(v);
            var q = PoissonSolve(div);

            for (int i = 0; i < q.GetLength(0); i++)
            {
                for (int j = 0; j < q.GetLength(1); j++)
                {
                    for (int k = 0; k < q.GetLength(2); k++)
                    {
                        q[i, j, k] *= -1;
                    }
                }
            }

            GaugeTransform(psi1, psi2, q);
        }


        //Упорядочевание
        public void Normalize(Complex[,,] psi1, Complex[,,] psi2)
        {
            for (int i = 0; i < psi1.GetLength(0); ++i)
            {
                for (int j = 0; j < psi1.GetLength(1); ++j)
                {
                    for (int k = 0; k < psi1.GetLength(2); ++k)
                    {
                        var psi_norm = Complex.Sqrt(Complex.Pow(Complex.Abs(psi1[i, j, k]), 2)
                                                    + Complex.Pow(Complex.Abs(psi2[i, j, k]), 2));
                        psi1[i, j, k] /= psi_norm;
                        psi2[i, j, k] /= psi_norm;
                    }
                }
            }
        }
    }

}