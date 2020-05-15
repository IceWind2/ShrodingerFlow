using System;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

using source.assets.Discrete_space;

namespace source.assets.Particles.utils
{
    public class VelocityHandler : Handler
    {
        private static int[] torus_res, torus_size;
        private static float[] torus_d;
        private static float _dt;
        private static float[] d_k1x, d_k1y, d_k1z, d_k2x, d_k2y, d_k2z, d_k3x, d_k3y, d_k3z, d_k4x, d_k4y, d_k4z;

        public static void init(ISF torus, int maxCnt)
        {
            _dt = torus.dt;

            torus_d = _gpu.CopyToDevice(new float[3] { torus.dx, torus.dy, torus.dz });
            torus_res = _gpu.CopyToDevice(new int[3] { torus.resx, torus.resy, torus.resz });
            torus_size = _gpu.CopyToDevice(new int[3] { torus.sizex, torus.sizey, torus.sizez });

            d_k1x = _gpu.Allocate<float>(maxCnt);
            d_k1y = _gpu.Allocate<float>(maxCnt);
            d_k1z = _gpu.Allocate<float>(maxCnt);

            d_k2x = _gpu.Allocate<float>(maxCnt);
            d_k2y = _gpu.Allocate<float>(maxCnt);
            d_k2z = _gpu.Allocate<float>(maxCnt);

            d_k3x = _gpu.Allocate<float>(maxCnt);
            d_k3y = _gpu.Allocate<float>(maxCnt);
            d_k3z = _gpu.Allocate<float>(maxCnt);

            d_k4x = _gpu.Allocate<float>(maxCnt);
            d_k4y = _gpu.Allocate<float>(maxCnt);
            d_k4z = _gpu.Allocate<float>(maxCnt);
        }

        public static void update_particles(float[,,] vx, float[,,] vy, float[,,] vz, int cnt)
        {
            int[] d_cnt = _gpu.CopyToDevice(new int[1] { cnt });
            float[] d_dt = _gpu.CopyToDevice(new float[1] { 0 });

            float[,,] d_vx = _gpu.CopyToDevice(vx);
            float[,,] d_vy = _gpu.CopyToDevice(vy);
            float[,,] d_vz = _gpu.CopyToDevice(vz);

            _gpu.Launch(cnt, 1, "staggered_velocity", x, y, z, d_k1x, d_k1y, d_k1z, d_dt, d_vx, d_vy, d_vz, d_k1x, d_k1y, d_k1z, torus_size, torus_res, torus_d);

            _gpu.Free(d_dt);
            d_dt = _gpu.CopyToDevice(new float[1] { _dt * (float)0.5 });

            _gpu.Launch(cnt, 1, "staggered_velocity", x, y, z, d_k1x, d_k1y, d_k1z, d_dt, d_vx, d_vy, d_vz, d_k2x, d_k2y, d_k2z, torus_size, torus_res, torus_d);

            _gpu.Launch(cnt, 1, "staggered_velocity", x, y, z, d_k2x, d_k2y, d_k2z, d_dt, d_vx, d_vy, d_vz, d_k3x, d_k3y, d_k3z, torus_size, torus_res, torus_d);

            _gpu.Free(d_dt);
            d_dt = _gpu.CopyToDevice(new float[1] { _dt });

            _gpu.Launch(cnt, 1, "staggered_velocity", x, y, z, d_k3x, d_k3y, d_k3z, d_dt, d_vx, d_vy, d_vz, d_k4x, d_k4y, d_k4z, torus_size, torus_res, torus_d);

            _gpu.Launch(cnt, 3, "update", x, y, z, d_k1x, d_k1y, d_k1z, d_k2x, d_k2y, d_k2z, d_k3x, d_k3y, d_k3z,
                d_k4x, d_k4y, d_k4z, d_cnt, d_dt);


            _gpu.Free(d_vx);
            _gpu.Free(d_vy);
            _gpu.Free(d_vz);
            _gpu.Free(d_cnt);
            _gpu.Free(d_dt);
        }   
    }
}
