namespace TinyTransformer.Core.Layers;

public class FeedForwardAuto : ILayer
{
    private readonly ILayer _l1;
    private readonly ILayer _l2;
    public FeedForwardAuto(int dModel, int hidden, Random rnd)
    {
        _l1 = new Linear(dModel, hidden, rnd);
        _l2 = new Linear(hidden, dModel, rnd);
    }

    public float[,] Forward(float[,] X)
    {
        var h = _l1.Forward(X);
        h = MathOps.ReLU(h); //non-linearity
        return _l2.Forward(h);
    }
}
