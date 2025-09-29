namespace TinyTransformer.Core;

//Linear layer: Y = X W + b
//core building block
//fully connected / dense
//Why we need a Linear class? -> Encapsulates a learnable affine transformation
public class Linear : ILayer
{
    private readonly float[,] _W;
    private readonly float[] _b;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="din">input dimensionality -> number of features per input sample</param>
    /// <param name="dout">output dimensionality -> number of neurons / outputs</param>
    /// <param name="rnd">random number generator to initialize weights</param>
    public Linear(int din, int dout, Random rnd)
    {
        float scale = (float)Math.Sqrt(2.0 / (din + dout));//helps keep activtions from exploding
        _W = MathOps.InitMatrix(din, dout, rnd, scale);
        _b = MathOps.InitVector(dout);
    }

    //deterministic
    public Linear(float[,] W, float[] b)
    {
        _W = W ?? throw new ArgumentNullException(nameof(W));
        _b = b ?? throw new ArgumentNullException(nameof(b));

        if (W.GetLength(1) != b.Length) 
            throw new ArgumentException("W (din x dout) and b (dout) must agree");
    }

    //how do I compute my outputs given my inputs and my current parameters
    public float[,] Forward(float[,] X)
    {
        var Y = MathOps.MatMul(X, _W);
        return MathOps.AddBias(Y, _b);
    }

}
