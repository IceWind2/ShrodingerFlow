using System;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using source.assets.Discrete_space;
using source.assets.Particles;

using source;

class TestCase
{
    static void Main(string[] args)
    {
        //PARAMETERS
        int[] vol_size = { 4, 2, 2 };      // box size
        int[] vol_res = { 64, 64, 64 };    // volume resolution
        float hbar = (float)0.1;           // Planck constant
        float dt = 1 / (float)48;          // time step
        var n_particles = 100;              // number of particles
        int max_iter = 20;
        int max_particles = n_particles * max_iter;     // max number of particles
        
        
        //INITIALISATION
        ISF.Init(vol_size, vol_res, hbar, dt);
        Particles.init(max_particles);
        var s = new Simulation(ISF.properties.resx, ISF.properties.resy, ISF.properties.resz, ISF.properties.hbar,
                                ISF.properties.px, ISF.properties.py, ISF.properties.pz);       
        Velocity vel = new Velocity(ISF.properties.resx, ISF.properties.resy, ISF.properties.resz);
        for (int i = 0; i < 10; i++)
        {
            ISF.Constraint(s);
        }
        
        cuFloatComplex[,,] ttt = new cuFloatComplex[vol_res[0], vol_res[1], vol_res[2]];
        ISF.psi1.CopyToHost(ttt);
        //MAIN ITERATION
        Stopwatch time = new Stopwatch();
        time.Start();
        
        Console.WriteLine("Start");
        for (int i = 0; i < max_iter; i++)
        {
            ISF.update_space(s);
            
            var tmp = s.generate_particles(n_particles);
            Particles.add_particles(tmp.x, tmp.y, tmp.z, n_particles);

            ISF.update_velocities(vel);
            
            Particles.calculate_movement(vel.vx, vel.vy, vel.vz);
        }

        

        float[] xx = Particles.x;
        float[] yy = Particles.y;
        float[] zz = Particles.z;

        for (int i = 0; i < 20; i++)
        {
            Console.Out.WriteLine(xx[i] + " " + yy[i] + " " + zz[i]);
        }
        
        time.Stop();
        TimeSpan ts = time.Elapsed;
        string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
        Console.WriteLine("RunTime " + elapsedTime);
    }

}
