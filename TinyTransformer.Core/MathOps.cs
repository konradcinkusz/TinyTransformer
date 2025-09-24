namespace TinyTransformer.Core;

public static class MathOps
{
    //Matrix multiplication
    public static float[,] MatMul(float[,] A, float[,] B)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);
        int p = B.GetLength(1);

        if (B.GetLength(0) != m) throw new ArgumentException("Incompatible size");

        var C = new float[n, p];

        for(int i = 0; i < n; i++)
        {
            for(int k = 0; k < m; k++)
            {
                var aik = A[i, k];
                for(int j=0;j< p; j++)
                {
                    C[i, j] += aik * B[i, j];
                }
            }
        }

        return C;
    }

    //Transpose martrix
    public static float[,] Transpose(float[,] A)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        var T = new float[m, n];

        for(int i = 0; i < n; i++)
        {
            for(int j= 0; j< m; j++)
            {
                T[j,i] = A[i, j];
            }
        }

        return T;
    }

    //ADD matrix (takes two 2D float arrays and returns
    //their element-wise sum
    //A and B must have the same shape (n x m)
    public static float[,] Add(float[,] A, float[,] B)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        if(B.GetLength(0) != n || B.GetLength(1) != m)
        {
            throw new ArgumentException("Shape mismatch");
        }

        var C = new float[n, m];

        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                C[i, j] = A[i, j] + B[i, j];
            }
        }

        return C;
    }
    
    //adding a bias vector to each row of a matrix
    //(a common step in neural networks)
    public static float[,] AddBias(float[,] A, float[] b)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        if (b.Length != m) 
            throw new ArgumentException("Bias length mismatch.");

        var C = new float[n, m];

        for(int i = 0; i< n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                C[i, j] = A[i, j] + b[j];
            }
        }

        return C;
    }

    //scalar–matrix multiplication method
    public static float[,] Scal(float[,]A, float s)
    {
        int
            n = A.GetLength(0);
        int m = A.GetLength(1);
        
        var C = new float[n, m];

        for( int i = 0; i< n; i++)
            for(int j = 0; j< m; j++)
            {
                C[i, j] = A[i, j] * s;
            }

        return C;
    }
}
