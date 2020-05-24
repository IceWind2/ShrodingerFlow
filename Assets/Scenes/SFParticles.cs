using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using ManagedCuda.VectorTypes;
using UnityEngine;
using source.assets.Discrete_space;
using source.assets.Particles;
using source;
using source.assets.CUDA_kernels;
using Debug = UnityEngine.Debug;

public class SFParticles : MonoBehaviour
{
    ParticleSystem.Particle[] cloud;
    bool bPointsUpdated = false;
    private ParticleSystem particleSystem;
    //PARAMETERS
    private  int[] vol_size = { 4, 2, 2 };      // box size
    private int[] vol_res = { 10, 10, 10 };    // volume resolution
    private float hbar = (float)0.1;           // Planck constant
    private float dt = 1 / (float)48;          // time step
    private const int n_particles = 200;              // number of particles
    private const int max_iter = 20;
    private int cur_iter = 0;
    private int max_particles = n_particles * max_iter;     // max number of particles
    private Simulation s;
    private Velocity vel;
    private cuFloatComplex[,,] ttt;
    void Start()
    {
        particleSystem = GetComponent<ParticleSystem>();
        //INITIALISATION
        ISF.Init(vol_size, vol_res, hbar, dt);
        Particles.init(max_particles);
        s = new Simulation(ISF.properties.resx, ISF.properties.resy, ISF.properties.resz, ISF.properties.hbar,
            ISF.properties.px, ISF.properties.py, ISF.properties.pz);
        vel = new Velocity(ISF.properties.resx, ISF.properties.resy, ISF.properties.resz);
        for (int i = 0; i < 10; i++)
        {
            ISF.Constraint(s);
        }

        ttt = new cuFloatComplex[vol_res[0], vol_res[1], vol_res[2]];
        ISF.psi1.CopyToHost(ttt);
    }
    
    void Update()
    {
        //MAIN ITERATION
        Stopwatch time = new Stopwatch();
        time.Start();

        Debug.Log("Start");
        ISF.update_space(s);

        if (cur_iter < max_iter)
        {
            var tmp = s.generate_particles(n_particles);
            Particles.add_particles(tmp.x, tmp.y, tmp.z, n_particles);
        }

        ISF.update_velocities(vel);

        Particles.calculate_movement(vel.vx, vel.vy, vel.vz);
        

        time.Stop();
        TimeSpan ts = time.Elapsed;
        string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
        Debug.Log("RunTime " + elapsedTime);

        DrawPoints();
        cur_iter += 1;
        new WaitForSeconds(0.6f);
    }

    public void DrawPoints()
    {
        cloud = new ParticleSystem.Particle[n_particles];

        float[] px = Particles.x;
        float[] py = Particles.y;
        float[] pz = Particles.z;

        //ToDO make color based on velocity
        //float[] vx = vel.vx;
        //float[] vy = vel.vy;
        //float[] vz = vel.vy;

        for (int ii = 0; ii < n_particles; ++ii)
        {
            var pos = new Vector3(px[ii], py[ii], pz[ii]);
            // var color = new Color(vx[ii], vy[ii], vz[ii]);
            cloud[ii].position = pos;
            cloud[ii].color = Color.red;
            cloud[ii].size = 0.1f;
        }
        
        particleSystem.SetParticles(cloud, cloud.Length);

    }

    public void OnEnable()
    {
        KernelLoader.init_loader();
    }
    public void OnDestroy()
    {
        KernelLoader.dispose_loader();
    }
}