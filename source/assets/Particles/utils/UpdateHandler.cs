using System;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace source.assets.Particles.utils
{
    public class RandomCoordinates
    {
        public float[] cos { get; set; }
        public float[] sin { get; set; }
    }

    public class UpdateHandler : Handler
    { 
        public static RandomCoordinates create_random(int cnt)
        {
            Random rnd = new Random();
            float[] t_cos = new float[cnt];
            float[] t_sin = new float[cnt];
            for (int i = 0; i < cnt; i++)
            {
                t_cos[i] = (float)Math.Cos(rnd.NextDouble() * 2 * Math.PI);
                t_sin[i] = (float)Math.Sin(rnd.NextDouble() * 2 * Math.PI);
            }
            return new RandomCoordinates { cos = t_cos, sin = t_sin };
        }

        public static void update_particles(float[] xx, float[] yy, float[] zz, int cnt, int size)
        {
            float[] d_xx = _gpu.CopyToDevice(xx);
            float[] d_yy = _gpu.CopyToDevice(yy);
            float[] d_zz = _gpu.CopyToDevice(zz);
            int[] d_size = _gpu.CopyToDevice(new int[1] { size });

            _gpu.Launch(cnt, 1, "add", x, y, z, d_xx, d_yy, d_zz, d_size);

            _gpu.Free(d_xx);
            _gpu.Free(d_yy);
            _gpu.Free(d_zz);
            _gpu.Free(d_size);
        }
    }
}
