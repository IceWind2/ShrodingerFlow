using System.Numerics;
using ManagedCuda.VectorTypes;
using UnityEngine;
using source.assets.Discrete_space;
using source.assets.Particles;
using source;
using Vector3 = UnityEngine.Vector3;

public class SFParticles : MonoBehaviour
{
    ParticleSystem.Particle[] cloud;
    bool bPointsUpdated = false;
    private ParticleSystem particleSystem;
    //PARAMETERS
    int[] vol_size = { 10, 5, 5 };      // box size
    int[] vol_res = { 128, 64, 64 };    // volume resolution
    float hbar = (float)0.1;           // Planck constant
    float dt = 1 / (float)12;          // time step
    int tmax = 85;
    float[] background_vel = {(float) -0.2, 0, 0};
        
    float r1 = (float)1.5; 
    float r2 = (float)0.9;              
    float[] n1 = {-1,0,0};         
    float[] n2 = {-1,0,0};

    int n_particles = 100000;

    Velocity vel;
    object lk= new object();

    UnityThreading.ActionThread myThread;
    void Start()
    {
        
        particleSystem = GetComponent<ParticleSystem>();
        myThread = UnityThreadHelper.CreateThread(Worker);
    }

    void Worker()
    {
        float[] cen1 = { vol_size[0] / 2f, vol_size[1] / 2f, vol_size[2] / 2f };
        float[] cen2 = { vol_size[0] / 2f, vol_size[1] / 2f, vol_size[2] / 2f };

        //INITIALISATION
        ISF.Init(vol_size, vol_res, hbar, dt);
        Particles.init(n_particles);

        //init psi
        float[] kvec = { background_vel[0] / hbar, background_vel[1] / hbar, background_vel[2] / hbar };
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
                    tmp1[i, j, k] = new cuFloatComplex((float)tmp.Real, (float)tmp.Imaginary);
                    tmp2[i, j, k] = new cuFloatComplex((float)(tmp.Real * 0.01), (float)(tmp.Imaginary * 0.01));
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
        System.Random rnd = new System.Random();
        for (int i = 0; i < n_particles; i++)
        {
            y[i] = (float)(rnd.NextDouble() * 4 + 0.5);
            z[i] = (float)(rnd.NextDouble() * 4 + 0.5);
            x[i] = 5;
        }

        Particles.add_particles(x, y, z, n_particles);

        vel = new Velocity(ISF.properties.resx, ISF.properties.resy, ISF.properties.resz);
        var c = new ParticleSystem.Particle[n_particles];

        while (true) {
            //MAIN ITERATION
            //incompressible Schroedinger flow
            ISF.update_space();

            //particle update
            ISF.update_velocities(vel);

            Particles.calculate_movement(vel);


            float[] px = Particles.x;
            float[] py = Particles.y;
            float[] pz = Particles.z;

            for (int ii = 0; ii < n_particles; ++ii)
            {
                var pos = new Vector3(px[ii], py[ii], pz[ii]);
                var lastPose = c[ii].position;
                c[ii].position = pos;
                c[ii].velocity = (pos - lastPose);
                var cc = c[ii].velocity.normalized;
                c[ii].color = new Color(cc.x, cc.y, cc.z);

                c[ii].size = 0.1f;
            }

            lock (pz) {
                cloud = c;
            }

            if (UnityThreading.ActionThread.CurrentThread.ShouldStop)
                return; // finish
        }

    }

    public void Update()
    {
        lock (lk) {
            if (cloud != null) {
                particleSystem.SetParticles(cloud, cloud.Length);
            }
        }
    }
}