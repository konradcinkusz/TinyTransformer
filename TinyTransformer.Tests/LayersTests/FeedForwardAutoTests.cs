namespace TinyTransformer.Tests.LayersTests;

public class FeedForwardAutoTests : TestsBase
{
    //1. positionwise property - if two input rows are identical, the coresponding
    // output rows must also be identical

    [Fact]
    public void FeedForwardAuto_IsPositionwise_RowsWithTheSameInputProducesSameOutput()
    {
        //Arrange
        int dModel = 4;
        int hidden = 8;
        int seed = 123;
        var rnd = new Random(seed);

        var ff = new FeedForwardAuto(dModel, hidden, rnd);

        //Build an input with duplicated rows
        var rowA = MathOps.InitVector(0.2f, -0.1f, 0.5f, 0.0f);
        var rowB = MathOps.InitVector(0.4f, -0.31f, 0.9f, -0.40f);
        var rowC = MathOps.InitVector(0.5f, -0.15f, -0.1f, 0.9f);

        var X = MathOps.InitMatrix(rowA, rowA, rowB, rowC);

        //Act 
        var Y = ff.Forward(X);

        //Assert - row 0 and 1 shold be nearly equal
        for (int i = 0; i < dModel; i++)
            Y[0, i].Should().BeApproximately(Y[1, i], 1e-5f); //compare each column value of 0 row with 1 row
    }
}
