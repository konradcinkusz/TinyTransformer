namespace TinyTransformer.Tests;

public sealed class ReLUMathOpsTest : MathOpsTests
{
    [Fact]
    public void ReLU_SetsNegativesToZero_AndKeppNonNegatives()
    {
        var A = new float[,]
        {
            {-2f, -0.5f, 0f, 1.2f },
            {3f, -7f, 5f, 4f }
        };

        var R = MathOps.ReLU(A);

        var expected = new float[,]
        {
            {0f, 0f, 0f, 1.2f },
            {3f, 0f, 5f, 4f }
        };

        MatricesShouldBeApproximatelyEqual(R, expected);
    }
}
