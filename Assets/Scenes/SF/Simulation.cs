using System;
using System.Numerics;
using ManagedCuda.VectorTypes;

namespace source
{
    public class Simulation
    {
        public readonly float[] JetVelocity = {1, 0, 0};
        
        public float[] nozzle_cen = { (float)(2 - 1.7), (float)(1 - 0.034), (float)(1 + 0.066) }; // nozzle center
        public float nozzle_len = (float)0.5;                   // nozzle length
        public float nozzle_rad = (float)0.5 ;                   // nozzle radius

        private bool[,,] isJet;
        private float[] kvec;
        private float omega;

        private float[,,] _px, _py, _pz;
        
        public Simulation(int resx, int resy, int resz, float hbar, float[,,] px, float[,,]py, float[,,] pz)
        {
            _px = px;
            _py = py;
            _pz = pz;
            
            isJet = new bool[resx, resy, resz];

            kvec = new float[] { JetVelocity[0] / hbar, JetVelocity[1] / hbar, JetVelocity[2] / hbar };
            omega = (JetVelocity[0] * JetVelocity[0] + JetVelocity[1] * JetVelocity[1] + JetVelocity[2] * JetVelocity[2]) / (2 * hbar);

            for (int i = 0; i < isJet.GetLength(0); i++)
            {
                for (int j = 0; j < isJet.GetLength(1); j++)
                {
                    for (int k = 0; k < isJet.GetLength(2); k++)
                    {
                        isJet[i, j, k] = (Math.Abs(px[i, j, k] - nozzle_cen[0]) <= nozzle_len / 2) &&
                                         ((Math.Pow(py[i, j, k] - nozzle_cen[1], 2) + (pz[i, j, k] - nozzle_cen[2]) * (pz[i, j, k] - nozzle_cen[2])) <= nozzle_rad * nozzle_rad);
                    }
                }
            }
        }
        
        public Coordinates generate_particles(int cnt)
        {
            var tmp = create_random(cnt);
            
            float[] xx = new float[cnt];
            float[] yy = new float[cnt];
            float[] zz = new float[cnt];

            for (int i = 0; i < cnt; i++)
            {
                xx[i] = nozzle_cen[0];
                yy[i] = (float)(nozzle_cen[1] + 0.9 * nozzle_rad * tmp.cos[i]);
                zz[i] = (float)(nozzle_cen[2] + 0.9 * nozzle_rad * tmp.sin[i]);
            }

            return new Coordinates(xx, yy, zz);
        }
        
        private static RandomCoordinates create_random(int cnt)
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
        
        private class RandomCoordinates
        {
            public float[] cos { get; set; }
            public float[] sin { get; set; }
        }

        public class Coordinates
        {
            public float[] x, y, z;

            public Coordinates(float[] xx, float[] yy, float[] zz)
            {
                x = xx;
                y = yy;
                z = zz;
            }
        }
    
        public void constraint(cuFloatComplex[,,] psi1, cuFloatComplex[,,] psi2, float t)
        {
            float phase;
            Complex amp1, amp2;
            Complex tmp1, tmp2;
            
            for (int i = 0; i < psi1.GetLength(0); i++)
            {
                for (int j = 0; j < psi1.GetLength(1); j++)
                {
                    for (int k = 0; k < psi1.GetLength(2); k++)
                    {
                        amp1 = Complex.Sqrt(psi1[i, j, k].real * psi1[i, j, k].real + psi1[i, j, k].imag * psi1[i, j, k].imag);
                        amp2 = Complex.Sqrt(psi2[i, j, k].real * psi2[i, j, k].real + psi2[i, j, k].imag * psi2[i, j, k].imag);
                        phase = kvec[0] * _px[i, j, k] + kvec[1] * _py[i, j, k] + kvec[2] * _pz[i, j, k] - omega * t;
                        if (isJet[i, j, k])
                        {
                            tmp1 = amp1 * Complex.Exp(Complex.ImaginaryOne * phase);
                            tmp2 = amp2 * Complex.Exp(Complex.ImaginaryOne * phase);
                            psi1[i, j, k] = new cuFloatComplex((float)tmp1.Real, (float)tmp1.Imaginary);
                            psi2[i, j, k] = new cuFloatComplex((float)tmp2.Real, (float)tmp2.Imaginary);
                            var rnd = new Random();
                            if (float.IsInfinity(psi1[i, j, k].real) || float.IsNaN(psi1[i, j, k].real))
                            {
                                psi1[i, j, k].real = (float)rnd.NextDouble() * 30;
                                psi1[i, j, k].imag = (float)rnd.NextDouble() * 30;
                            }
                            
                            if (float.IsInfinity(psi2[i, j, k].real) || float.IsNaN(psi2[i, j, k].real))
                            {
                                psi2[i, j, k].real = (float)rnd.NextDouble() * 3;
                                psi2[i, j, k].imag = (float)rnd.NextDouble() * 3;
                            }
                        }
                    }
                }
            }
        }
    }
}