using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Numerics;
using System.Text;
using ManagedCuda;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using source.assets.Discrete_space;
using source.assets.Particles;

using source;

class TestCase
{
    static void iterate(int[] vol_res, int n_particles)
    {
        int[] vol_size = { 10, 5, 5 };      // box size
        float hbar = (float)0.1;           // Planck constant
        float dt = 1 / (float)24;          // time step
        int tmax = 85;
        float[] background_vel = {(float) -0.2, 0, 0};
        
        float r1 = (float)1.5; 
        float r2 = (float)0.9;              
        float[] n1 = {-1,0,0};         
        float[] n2 = {-1,0,0};        

        float[] cen1 = {vol_size[0] / 2f, vol_size[1] / 2f, vol_size[2] / 2f}; 
        float[] cen2 = {vol_size[0] / 2f, vol_size[1] / 2f, vol_size[2] / 2f}; 
 
        
        //INITIALISATION
        ISF.Init(vol_size, vol_res, hbar, dt);
        Particles.init(n_particles);
        
        //init psi
        float[] kvec = {background_vel[0] / hbar, background_vel[1] / hbar, background_vel[2] / hbar};
        float phase;
        var tmp1 = new cuFloatComplex[ISF.properties.resx, ISF.properties.resy, ISF.properties.resz];
        var tmp2 = new cuFloatComplex[ISF.properties.resx, ISF.properties.resy, ISF.properties.resz];
        Complex tmp;
        for (int i = 0; i < vol_res[0]; i++)
        {
            for (int j = 0; j < vol_res[1]; j++)
            {
                for (int k = 0; k < vol_res[2]; k++)
                {
                    phase = kvec[0] * ISF.properties.px[i, j, k] +
                            kvec[1] * ISF.properties.py[i, j, k] +
                            kvec[2] * ISF.properties.pz[i, j, k];
                    tmp = Complex.Exp(Complex.ImaginaryOne * phase);
                    tmp1[i, j, k] = new cuFloatComplex((float) tmp.Real, (float) tmp.Imaginary);
                    tmp2[i, j, k] = new cuFloatComplex((float) (tmp.Real * 0.01), (float) (tmp.Imaginary * 0.01));
                }
            }
        }
        float d = ISF.properties.dx * 5;
        
        ISF.add_circle(tmp1, cen1, n1, r1, d);
        ISF.add_circle(tmp1, cen2, n2, r2, d);
        
        ISF.psi1.CopyToDevice(tmp1);
        ISF.psi2.CopyToDevice(tmp2);

        ISF.Normalize();
        ISF.PressureProject();
        
        //init particles
        var x = new float[n_particles];
        var y = new float[n_particles];
        var z = new float[n_particles];
        Random rnd = new Random();
        for (int i = 0; i < n_particles; i++)
        {
            y[i] = (float)(rnd.NextDouble() * 4 + 0.5);
            z[i] = (float)(rnd.NextDouble() * 4 + 0.5);
            x[i] = 5;
        }
        
        Particles.add_particles(x, y, z, n_particles);
        
        Velocity vel = new Velocity(ISF.properties.resx, ISF.properties.resy, ISF.properties.resz);
        
        
        Stopwatch time = new Stopwatch();
        time.Start();
        //MAIN ITERATION
        int itermax = 500;
        for (int i = 0; i < itermax; i++)
        {
            //incompressible Schroedinger flow
            ISF.update_space();
            
            //particle update
            ISF.update_velocities(vel);

            Particles.calculate_movement(vel);
        }

        time.Stop();
        TimeSpan ts = time.Elapsed;
        string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
        using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(@"./test.txt", true))
        {
            file.WriteLine(elapsedTime);
        }
    }

    static void Main(string[] args)
    {
        int res = 16;
        int num = 1000;
        for (int i = 0; i < 3; i++)
        {
            iterate(new int[] { res, res / 2, res / 2 }, num);
            iterate(new int[] { res, res / 2, res / 2 }, num * 5);
            iterate(new int[] { res, res / 2, res / 2 }, num * 10);
            iterate(new int[] { res, res / 2, res / 2 }, num * 20);
	    iterate(new int[] { res, res / 2, res / 2 }, num * 50);
            res *= 2;
        }
        
    }

}
