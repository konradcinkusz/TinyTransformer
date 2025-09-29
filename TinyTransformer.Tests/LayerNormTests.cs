namespace TinyTransformer.Tests;

public class LayerNormTests : MathOpsTests
{
    [Fact]
    public void LayerNorm_NormalizeEachRow()
    {
        var X = new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } };

        var ln = new LayerNorm(3);
        var Y = ln.Forward(X);

        for(int i = 0; i< Y.GetLength(0); i++)
        {
            int d = Y.GetLength(1);
            var m = MathOps.Mean(Y, i, d);
            var v = MathOps.Variance(Y, i, d, m);
            m.Should().BeApproximately(0f, 1e-4f);
            v.Should().BeApproximately(1f, 1e-4f);
        }
    }
}
