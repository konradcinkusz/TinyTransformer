namespace TinyTransformer.Core;

//It keeps transformers block's activations and gradients stable
//LayerNorm compute mean and variance, normalize the row then scale and shift it with
//learned vector gamma and beta
public class LayerNorm : ILayer
{
    private readonly int _d; // feature dimension (length of the vector that represents a single token)
    //in Transformer it's the size of each token's emvedding or hidden state
    private readonly float[] _gamma, _beta;
    private const float Eps = 1e-5f;
    public LayerNorm(int d)
    {
        _d = d;
        _gamma = Enumerable.Repeat(1f, d).ToArray();
        _beta = Enumerable.Repeat(0f, d).ToArray();
    }

    public float[,] Forward(float[,] X)
    {
        int n = X.GetLength(0); // n = number of tokens (rows)
        var Y = new float[n, _d]; //output buffer, same shape as X

        for (int i = 0; i < n; i++) //process each token independently
        {
            float mean = MathOps.Mean(X, i, _d);
            float variance = MathOps.Variance(X, i, _d, mean);
            //the core math of layer normalization

            //CORE LOGIC
            //1. compute the inverse standard deviation
            float inv = 1f / (float)Math.Sqrt(variance + Eps);

            //2. normalize each feature, then scale and shift - loop
            for(int j = 0; j < _d; j++)
            {
                float norm = (X[i,j] - mean) * inv;
                Y[i,j] = _gamma[j]* norm + _beta[j];
            }
        }

        return Y;
    }
}
