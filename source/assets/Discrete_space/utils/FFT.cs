using System.Numerics;

namespace source.assets.Discrete_space.utils
{
    static class FFT
    {
        /*public static void configure(GPGPU gpu)
        {
            _GPU = gpu;

            _fftGPU = Cudafy.Maths.FFT.GPGPUFFT.Create(gpu);
        }

        public static void fft_c(Complex[,,] data, Complex[,,] result, int[] dim, bool inv)
        {
            var fftPlan = _fftGPU.Plan3D(Cudafy.Maths.FFT.eFFTType.Complex2Complex, Cudafy.Maths.FFT.eDataType.Single,
                                    dim[0], dim[1], dim[2], 1);

            fftPlan.Execute(data, result, inv);
        }

        public static void fft_r(float[,,] data, Complex[,,] result, int[] dim,  bool inv)
        {
            var fftPlan = _fftGPU.Plan3D(Cudafy.Maths.FFT.eFFTType.Real2Complex, Cudafy.Maths.FFT.eDataType.Single,
                                    dim[0], dim[1], dim[2], 1);

            fftPlan.Execute(data, result, inv);
        }

        public static void shift(Complex[,,] data, int[] dim)
        {
            _GPU.Launch(1, new dim3(dim[0], dim[1], dim[2]), "fftShift", data);
        }

        public static void mul_each(Complex[,,] data, Complex[,,] mask, int[] dim)
        {
            _GPU.Launch(1, new dim3(dim[0], dim[1], dim[2]), "MulEach_c", data, mask);
        }

        public static void mul_each(float[,,] data, float[,,] mask, int[] dim)
        {
            _GPU.Launch(1, new dim3(dim[0], dim[1], dim[2]), "MulEach_r", data, mask);
        }

        [Cudafy]
        private static void MulEach_c(GThread thread, Complex[,,] data, Complex[,,] coefs)
        {
            int i = thread.threadIdx.x;
            int j = thread.threadIdx.y;
            int k = thread.threadIdx.z;

            data[i, j, k] *= coefs[i, j, k];
        }

        [Cudafy]
        private static void MulEach_r(GThread thread, float[,,] data, float[,,] coefs)
        {
            int i = thread.threadIdx.x;
            int j = thread.threadIdx.y;
            int k = thread.threadIdx.z;

            data[i, j, k] *= coefs[i, j, k];
        }

        [Cudafy]
        private static void fftShift(GThread thread, Complex[,,] x)
        {
            int i = thread.threadIdx.x;
            int j = thread.threadIdx.y;
            int k = thread.threadIdx.z;

            int iStep = x.GetLength(0) / 2;
            int jStep = x.GetLength(1) / 2;
            int kStep = x.GetLength(2) / 2;

            Complex tmp = x[i, j, k];
            x[i, j, k] = x[i + iStep, j + jStep, k + kStep];
            x[i + iStep, j + jStep, k + kStep] = tmp;
        }*/
    }
}
