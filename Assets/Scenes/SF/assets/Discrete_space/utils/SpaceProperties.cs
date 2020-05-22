namespace source.assets.Discrete_space.utils
{
    public class SpaceProperties
    {
         public float[,,] px, py, pz;         // px::Array{float64,3}; py::Array{float64,3}; pz::Array{float64,3}  coordinates of grid points
         public float dx, dy, dz;                    // edge length
         public int sizex, sizey, sizez;             // size of grid
         public int resx, resy, resz;             // number of grid points in each dimension (Количество точек сетки в каждом измерении)
         public int num;
         public float dt;                       // time step
         public float hbar;                  // reduced Planck constant (Понижение постоянной Планка)

         public SpaceProperties(int[] vol_size, int[] vol_res, float HBAR, float DT)
         {
             sizex = vol_size[0];
             sizey = vol_size[1];
             sizez = vol_size[2];

             resx = vol_res[0];
             resy = vol_res[1];
             resz = vol_res[2];
             num = resx * resy * resz;
             
             dx = sizex / (float)resx;
             dy = sizey / (float)resy;
             dz = sizez / (float)resz;
             
             hbar = HBAR;
             dt = DT;
         }
    }
}