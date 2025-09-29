namespace TinyTransformer.Tests;

public class LinearTests : MathOpsTests
{
    [Fact]
    public void Forward_SmallExample()
    {
        var X = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var W = new float[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } };
        var b = new float[] { 10, 20 };

        var lin = new Linear(W, b);
        var Y = lin.Forward(X);

        //row0 [1,2,3]*W + b
        //row1 [4,5,6]*W + b
        var expected = new float[,]
        {
            { 14, 25},
            { 20, 31 },
        };

        MatricesShouldBeApproximatelyEqual(Y, expected);
    }

    [Fact]
    public void Forward_ZeroBias_EqualsPureMatMul()
    {

        var X = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var W = new float[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } };
        var b = new float[] { 0f, 0f };

        var lin = new Linear(W, b);
        var Y = lin.Forward(X);

        var expected = MathOps.MatMul(X, W);

        MatricesShouldBeApproximatelyEqual(Y, expected);
    }

    [Fact]
    public void Foreward_CorrectOutputShape()
    {
        var X = new float[5, 3];
        var W = new float[3, 4]; 
        var b = new float[4];

        var lin = new Linear(W, b);
        var Y = lin.Forward(X);//expecting shape 5x4

        Y.GetLength(0).Should().Be(5); //batch
        Y.GetLength(1).Should().Be(4); //dout
    }

    [Fact]
    public void Foreward_InputMismatch_Throws()
    {
        var X = new float[5, 3];
        var W = new float[4, 3];
        var b = new float[3];

        var lin = new Linear(W, b);
        Action act = () => lin.Forward(X);

        act.Should().Throw<ArgumentException>();
    }
}
