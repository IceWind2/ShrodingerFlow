using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace source.assets.utils
{
    public static class Matlab
    {
        public class ThreeDIndices
        {
            public int[,,] x, y, z;

            public ThreeDIndices(int xLength, int yLength, int zLength)
            {
                x = new int[xLength, yLength, zLength];
                y = new int[xLength, yLength, zLength];
                z = new int[xLength, yLength, zLength];
            }
        }
        public static ThreeDIndices ndgrid(int[] x, int[] y, int[] z)
        {
            var result = new ThreeDIndices(x.Length, y.Length, z.Length);
            //TODO: править баг когда z != x != y
            int xi = 0, yi = 0, zi = 0;
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < y.Length; j++)
                {
                    for (int k = 0; k < z.Length; k++)
                    {
                        result.x[i, j, k] = x[xi];
                        result.y[i, j, k] = y[yi];
                        result.z[i, j, k] = z[zi++];
                    }
                    yi++;
                    zi = 0;
                }
                yi = 0;
                xi++;
            }

            return result;
        }

        public static void ForEach<T>(this IEnumerable<T> data, Action<T> act)
        {
            foreach (var e in data)
            {
                act(e);
            }
        }

        public static IEnumerable<T> Select<T>(this IEnumerable<T> data, Func<int, T, T> act)
        {
            var result = data.ToList();
            for (int i = 0; i < result.Count; ++i)
            {
                result[i] = act(i, data.ElementAt(i));
            }

            return result;
        }

        public static OutT[,,] Select3D<T, OutT>(this T[,,] data, Func<T, int, int, int, OutT> act)
        {
            var result = new OutT[data.GetLength(0), data.GetLength(1), data.GetLength(2)];
            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    for (int k = 0; k < data.GetLength(2); k++)
                    {
                        result[i, j, k] = act(data[i, j, k], i, j, k);
                    }
                }
            }

            return result;
        }

        public static OutT[][][] Select3D<T, OutT>(this T[][][] data, Func<T, int, int, int, OutT> act)
        {
            var result = new OutT[data.Length][][];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = new OutT[data[0].Length][];
                for (int j = 0; j < data[0].Length; j++)
                {
                    result[i][j] = new OutT[data[0][0].Length];
                    for (int k = 0; k < data[0][0].Length; k++)
                    {
                        result[i][j][k] = act(data[i][j][k], i, j, k);
                    }
                }
            }

            return result;
        }
    }
}
