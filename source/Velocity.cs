using ManagedCuda;

namespace source
{
    public class Velocity
    {
        public CudaDeviceVariable<float> vx;
        public CudaDeviceVariable<float> vy;
        public CudaDeviceVariable<float> vz;

        public Velocity(int rx, int ry, int rz)
        {
            var tmp = rx * ry * rz;
            vx = new CudaDeviceVariable<float>(tmp);
            vy = new CudaDeviceVariable<float>(tmp);
            vz = new CudaDeviceVariable<float>(tmp);
        }
    }
}