namespace TinyTransformer.Core.Layers;

public class TransformerEncoderBlock : ILayer
{
    private readonly ILayer _selfAttention;
    private readonly ILayer _feedForward;
    private readonly ILayer _ln1;
    private readonly ILayer _ln2;

    public TransformerEncoderBlock(int dModel, int dK, int ffHidden, Random rnd)
    {
        _selfAttention = new SelfAttention(dModel, dK, rnd);
        _feedForward = new FeedForwardAuto(dModel, ffHidden, rnd);
        _ln1 = new LayerNorm(dModel);
        _ln2 = new LayerNorm(dModel);
    }

    public float[,] Forward(float[,] X)
    {
        var attentionOutput = _selfAttention.Forward(X);
        var x1 = _ln1.Forward(MathOps.Add(X, attentionOutput));
        var ffOutput = _feedForward.Forward(x1);
        return _ln2.Forward(MathOps.Add(x1, ffOutput));
    }
}
