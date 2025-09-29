namespace TinyTransformer.Tests.MathOpsTests;

public class BasicTestsBase : TestsBase
{
    [Fact]
    public void MatMul_SmallExample_Works()
    {
        // A: 2x3, B: 3x2 -> C: 2x2
        var A = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var B = new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } };

        var C = MathOps.MatMul(A, B);

        var expected = new float[,] {
            { 58, 64 },        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
            { 139, 154 }       // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
        };

        MatricesShouldBeApproximatelyEqual(C, expected);
    }

    [Fact]
    public void Transpose_Works()
    {
        var A = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }; // 2x3
        var AT = MathOps.Transpose(A);                     // 3x2

        var expected = new float[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } };
        MatricesShouldBeApproximatelyEqual(AT, expected);
    }

    [Fact]
    public void Add_Works()
    {
        var A = new float[,] { { 1, 2 }, { 3, 4 } };
        var B = new float[,] { { 5, 6 }, { 7, 8 } };

        var C = MathOps.Add(A, B);

        var expected = new float[,] { { 6, 8 }, { 10, 12 } };
        MatricesShouldBeApproximatelyEqual(C, expected);
    }

    [Fact]
    public void AddBias_AddsVectorToEachRow()
    {
        var A = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }; // 2x3
        var b = new float[] { 10, 20, 30 };                // length 3

        var C = MathOps.AddBias(A, b);

        var expected = new float[,] { { 11, 22, 33 }, { 14, 25, 36 } };
        MatricesShouldBeApproximatelyEqual(C, expected);
    }

    [Fact]
    public void Scale_Works()
    {
        var A = new float[,] { { 1, -2 }, { 0.5f, 4 } };
        var C = MathOps.ScalarMatrixMultiplication(A, 2f);

        var expected = new float[,] { { 2, -4 }, { 1f, 8 } };
        MatricesShouldBeApproximatelyEqual(C, expected);
    }

    // ----------------- error cases -----------------

    [Fact]
    public void MatMul_DimensionMismatch_Throws()
    {
        var A = new float[,] { { 1, 2 }, { 3, 4 } };       // 2x2
        var B = new float[,] { { 1, 2, 3 } };              // 1x3 (mismatch)

        Action act = () => MathOps.MatMul(A, B);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Add_DimensionMismatch_Throws()
    {
        var A = new float[,] { { 1, 2 }, { 3, 4 } };
        var B = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };

        Action act = () => MathOps.Add(A, B);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void AddBias_LengthMismatch_Throws()
    {
        var A = new float[,] { { 1, 2, 3 } };
        var b = new float[] { 1, 2 };

        Action act = () => MathOps.AddBias(A, b);
        act.Should().Throw<ArgumentException>();
    }

}
