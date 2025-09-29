namespace TinyTransformer.Tests;

public class InitMatrixTests : MathOpsTests
{
    [Fact]
    public void InitMatrix_GenerateCorrectShape_AndValuesWithinRange()
    {
        int rows = 3, cols = 5;
        float scale = 0.04f; //[-0.04f, 0.04)
        var rnd = new Random(123);

        var M = MathOps.InitMatrix(rows, cols, rnd, scale);

        M.GetLength(0).Should().Be(rows);
        M.GetLength(1).Should().Be(cols);

        for(int i=0;i<rows;i++)
            for(int j= 0;j<cols;j++)
            {
                M[i, j].Should().BeGreaterThanOrEqualTo(-scale);
                M[i, j].Should().BeLessThan(scale);
            }
    }
}
