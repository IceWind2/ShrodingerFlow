using System;
using System.Diagnostics;
using System.Numerics;
using ManagedCuda.BasicTypes;
using source.assets.Discrete_space;
using source.assets.Particles;

using  ManagedCuda;

class TestCase
{
    private static bool[,,] isJet;

    private static float[] kvec;

    private static float omega;

    public static void constraint(ISF isf, Complex[,,] psi1, Complex[,,] psi2, float t = 0)
    {
        var phase = new float[isf.resx, isf.resy, isf.resz];

        Complex[][][] amp1 = new Complex[isf.resx][][];
        Complex[][][] amp2 = new Complex[isf.resx][][];

        for (int i = 0; i < isf.resx; i++)
        {
            amp1[i] = new Complex[isf.resy][];
            amp2[i] = new Complex[isf.resy][];
            for (int j = 0; j < isf.resy; j++)
            {
                amp1[i][j] = new Complex[isf.resz];
                amp2[i][j] = new Complex[isf.resz];
                for (int k = 0; k < isf.resz; k++)
                {
                    amp1[i][j][k] = Complex.Abs(psi1[i, j, k]);
                    amp2[i][j][k] = Complex.Abs(psi2[i, j, k]);
                    phase[i, j, k] = kvec[0] * isf.px[i, j, k] + kvec[1] * isf.py[i, j, k] + kvec[2] * isf.pz[i, j, k] - omega * t;
                    if (isJet[i, j, k])
                    {
                        psi1[i, j, k] = amp1[i][j][k] * Complex.Exp(Complex.ImaginaryOne * phase[i, j, k]);
                        psi2[i, j, k] = amp2[i][j][k] * Complex.Exp(Complex.ImaginaryOne * phase[i, j, k]);
                    }
                }
            }
        }

        isf.PressureProject(psi1, psi2);
    }

    static void Main(string[] args)
    {
        //PARAMETERS
        int[] vol_size = { 4, 2, 2 };   // box size
        int[] vol_res = { 10, 10, 10 }; // volume resolution
        float hbar = (float)0.1;           // Planck constant
        float dt = 1 / (float)48;             // time step
        int tmax = 1;

        float[] jet_velocity = { 1, 0, 0 }; // jet velocity

        float[] nozzle_cen = { (float)(2 - 1.7), (float)(1 - 0.034), (float)(1 + 0.066) }; // nozzle center
        float nozzle_len = (float)0.5;                   // nozzle length
        float nozzle_rad = (float)0.5 ;                   // nozzle radius

        var n_particles = 50;   // number of particles
        const int max_particles = 200;
        

        //INITIALISATION
        ISF isf = new ISF(vol_size, vol_res, hbar, dt);
        isf.hbar = hbar;
        isf.dt = dt;

        Particles.init(max_particles, isf);

        

        // Set nozzle and initialize psi
        Complex[,,] psi1 = new Complex[isf.resx, isf.resy, isf.resz];
        Complex[,,] psi2 = new Complex[isf.resx, isf.resy, isf.resz];

        isJet = new bool[isf.resx, isf.resy, isf.resz];

        kvec = new float[] { jet_velocity[0] / isf.hbar, jet_velocity[1] / isf.hbar, jet_velocity[2] / isf.hbar };
        omega = (jet_velocity[0] * jet_velocity[0] + jet_velocity[1] * jet_velocity[1] + jet_velocity[2] * jet_velocity[2]) / (2 * isf.hbar);

        for (int i = 0; i < isJet.GetLength(0); i++)
        {
            for (int j = 0; j < isJet.GetLength(1); j++)
            {
                for (int k = 0; k < isJet.GetLength(2); k++)
                {
                    isJet[i, j, k] = (Math.Abs(isf.px[i, j, k] - nozzle_cen[0]) <= nozzle_len / 2) &&
                                     ((Math.Pow(isf.py[i, j, k] - nozzle_cen[1], 2) + (isf.pz[i, j, k] - nozzle_cen[2]) * (isf.pz[i, j, k] - nozzle_cen[2])) <= nozzle_rad * nozzle_rad);
                    psi1[i, j, k] = 1;
                    psi2[i, j, k] = 0.01;
                }
            }
        }

        isf.Normalize(psi1, psi2);

        // constrain velocity
        for (int i = 0; i < 10; i++)
        {
            constraint(isf, psi1, psi2, 0);
        }

        //MAIN ITERATION
        ISF.Velocity vel = new ISF.Velocity();
        var itermax = Math.Ceiling(tmax / dt);

        Stopwatch time = new Stopwatch();
        time.Start();

        Console.WriteLine("Start");
        for (int iter = 0; iter < 1; iter++)
        {
            var t = iter * dt;

            // incompressible Schroedinger flow
            isf.schroedinger_flow(psi1, psi2);
            isf.Normalize(psi1, psi2);
            isf.PressureProject(psi1, psi2);

            //constrain velocity
            constraint(isf, psi1, psi2, t);

            //particle birth
            Particles.add_particles(n_particles, nozzle_cen, nozzle_rad);
            
            //advect and show particles
            vel = isf.velocity_oneForm(psi1, psi2, isf.hbar);
            vel = isf.staggered_sharp(vel.vx, vel.vy, vel.vz);

            Particles.calculate_movement(vel.vx, vel.vy, vel.vz);
        }

        float[] xx = Particles.x;
        float[] yy = Particles.y;
        float[] zz = Particles.z;

        for (int i = 0; i < 10; i++)
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
