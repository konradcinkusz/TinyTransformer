namespace TinyTransformer.Tests.MathOpsTests;

public sealed class SoftMaxTestsBase : TestsBase
{
    //1. Uniform distribution case
    [Fact]
    public void SoftmaxRows_UniformRows_ReturnUniformProbabilities()
    {
        var A = new float[,]
        {
            { 0f, 0f, 0f}, //when all entries in the row are the same
            { 5f, 5f, 5f} //each entry becomes 1 / m
        };

        var S = MathOps.SoftmaxRows(A);

        var oneThird = 1f / 3f;
        var expected = new float[,]
            { 
                {oneThird,oneThird,oneThird },
                {oneThird,oneThird,oneThird }, 
            };

        MatricesShouldBeApproximatelyEqual(S, expected);
    }

    private static float[,] A = new float[,] { { 1000f, 1001f, 999f } };

    //2. checks the numerical stability with large logits
    // large logits = very large raw model outputs
    // logit = raw, unnomralized score 
    [Fact]
    public void SoftmaxRows_LargeValues_NumericallyStable()
    {
        var S = MathOps.SoftmaxRows(A);

        var E = new float[,] { { -1, 0, -2 } };

        var expectedS = MathOps.SoftmaxRows(E);

        MatricesShouldBeApproximatelyEqual(S, expectedS);
    }

    //3. Each row sums to approx 1
    [Fact]
    public void SoftmaxRows_AddToOne()
    {
        var S = MathOps.SoftmaxRows(A);
        int n = S.GetLength(0);
        int m = S.GetLength(1);

        for(int i = 0; i < n; i++)
        {
            float sum = 0f;

            for(int j = 0; j<m; j++)
            {
                S[i, j].Should().BeGreaterThan(0f).And.BeLessThan(1f);
                sum += S[i, j];
            }

            sum.Should().BeApproximately(1f,1e-5f);
        }
    }
}
