namespace TinyTransformer.Tests;

public class TestsBase
{
    protected static void MatricesShouldBeApproximatelyEqual(float[,] actual, float[,] expected, float tol = 1e-5f)
    {
        actual.GetLength(0).Should().Be(expected.GetLength(0));
        actual.GetLength(1).Should().Be(expected.GetLength(1));
        for (int i = 0; i < actual.GetLength(0); i++)
            for (int j = 0; j < actual.GetLength(1); j++)
                actual[i, j].Should().BeApproximately(expected[i, j], tol);
    }
    protected static float[] RandomVector(int d, Random rnd)
    {
        var v = new float[d];
        for (int i = 0; i < d; i++)
            v[i] = (float)rnd.NextDouble() - 0.5f;
        return v;
    }

    protected static float[,] TakeRow(float[] row, int nCopies)
    {
        var X = new float[nCopies, row.Length];
        for (int i = 0; i < nCopies; i++)
            for (int j = 0; j < row.Length; j++)
                X[i, j] = row[j];
        return X;
    }
}
