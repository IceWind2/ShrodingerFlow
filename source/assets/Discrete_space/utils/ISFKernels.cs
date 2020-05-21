using ManagedCuda;
using ManagedCuda.VectorTypes;
using source.assets.CUDA_kernels;

namespace source.assets.Discrete_space.utils
{
    public static class ISFKernels
    {
        public static CudaKernel staggered, div, velocity_one, normalize, gauge, shift, mul_each, copy;

        public static void Init(SpaceProperties properties)
        {
            staggered = KernelLoader.load_kernel("staggered_sharp");
            staggered.BlockDimensions = new dim3(1, 1, 1);
            staggered.GridDimensions = new dim3(properties.num, 1, 1);

            div = KernelLoader.load_kernel("div");
            div.BlockDimensions = new dim3(1, 1, 1);
            div.GridDimensions = new dim3(properties.num, 1, 1);

            velocity_one = KernelLoader.load_kernel("velocity_one");
            velocity_one.BlockDimensions = new dim3(1, 1, 1);
            velocity_one.GridDimensions = new dim3(properties.num, 1, 1);
            
            normalize = KernelLoader.load_kernel("normalize");
            normalize.BlockDimensions = new dim3(1, 1, 1);
            normalize.GridDimensions = new dim3(properties.num, 1, 1);
            
            gauge = KernelLoader.load_kernel("gauge");
            gauge.BlockDimensions = new dim3(1, 1, 1);
            gauge.GridDimensions = new dim3(properties.num, 1, 1);
            
            shift = KernelLoader.load_kernel("shift");
            shift.BlockDimensions = new dim3(1, 1, 1);
            shift.GridDimensions = new dim3(properties.num, 1, 1);
            
            mul_each = KernelLoader.load_kernel("mul_each");
            mul_each.BlockDimensions = new dim3(1, 1, 1);
            mul_each.GridDimensions = new dim3(properties.num, 1, 1);
            
            copy = KernelLoader.load_kernel("copy");
            copy.BlockDimensions = new dim3(1, 1, 1);
            copy.GridDimensions = new dim3(properties.num, 1, 1);
        }
    }
}