namespace TinyTransformer.Tests.LayersTests;

public class SelfAttentionTests : TestsBase
{
    [Fact]
    public void SelfAttention_WithIdenticalTokens_ProducesIdenticalOutputPerRow()
    {
        //Arrange
        int T = 5;
        int dModel = 8;
        int dK = 4;
        var rnd = new Random(123);

        //creates a single random vector of lenght dModel
        //one token embedding [x1,x2,....,x_dModel
        var row = RandomVector(dModel, rnd);
        var X = TakeRow(row, T); //[T x dModel] all rows equal

        var attention = new SelfAttention(dModel, dK, rnd);

        //Act
        var Y = attention.Forward(X); //[T x dModel]

        //Assert: every row of Y should be (approximately) identical
        for(int i  = 1; i< T; i++)
        {
            int m = Y.GetLength(1);
            for (int j = 0; j < m; j++)
                Y[i, j].Should().BeApproximately(Y[0, j], 1e-5f);
        }
    }
}
