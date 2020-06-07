using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using source.assets.Particles.utils;
using System.Collections.Generic;
using source.assets.Discrete_space;
using System;
using System.Numerics;

namespace source.assets.Particles
{
    public static class Particles
    {
        private static List<ParticleSet> particles;
        private static int _size;
        public static CudaDeviceVariable<float> x, y, z;

        public static void init(int max_particles)
        {
            x = new CudaDeviceVariable<float>(max_particles);
            y = new CudaDeviceVariable<float>(max_particles);
            z = new CudaDeviceVariable<float>(max_particles);
            
            _size = 0;

            Handler.set_particles(x, y, z);

            UpdateHandler.init();

            VelocityHandler.init(max_particles);
        }

        public static void add_particles(float[] xx, float[] yy, float[] zz, float[] vx, float[] vy, float[] vz)
        {
            int cnt = xx.Length;

            var tmp1 = new cuFloatComplex[ISF.properties.resx, ISF.properties.resy, ISF.properties.resz];

            ISF.psi1.CopyToHost(tmp1);
            int curx, cury, curz;
            Complex tmp;

            for (int i = 0; i < cnt; i++)
            {
                curx = (int)Math.Round(xx[i] / ISF.properties.dx);
                cury = (int)Math.Round(yy[i] / ISF.properties.dy);
                curz = (int)Math.Round(zz[i] / ISF.properties.dz);

                tmp = new Complex(tmp1[curz, cury, curz].real, tmp1[curz, cury, curz].imag);
                tmp *= Complex.Exp(Complex.ImaginaryOne * vx[i]);
                tmp1[curx, cury, curz] = new cuFloatComplex((float)tmp.Real, (float)tmp.Imaginary);
            }

            ISF.psi1.CopyToDevice(tmp1);

            UpdateHandler.update_particles(xx, yy, zz, cnt, _size);

            particles.Add(new ParticleSet(xx, yy, zz, vx, vy, vz));

            _size += cnt;
        }

        public static void calculate_movement(Velocity vel)
        {
            float[] tmpv = vel.vx;
            int tmps = 0;
            for (int i = 0; i < particles.Count; i++)
            {
                float[] curv = particles[i].vx;
                for (int j = 0; j < particles[i].size; j++)
                {
                    tmpv[tmps + j] += curv[j];
                }
                tmps += particles[i].size;
            }
            vel.vx = tmpv;

            VelocityHandler.update_particles(vel.vx, vel.vy, vel.vz, _size);
        }
    }
}