namespace TinyTransformer.Tests.MathOpsTests;

public class InitVectorTests : TestsBase
{
    [Fact]
    public void InitVector_CreatesVectorWithCorrectLenghtAndDeafultZeroValues()
    {
        int n = 5; 
        
        var v = MathOps.InitVector(n);

        v.Length.Should().Be(n);
        for(int i = 0; i < n; i++)
            v[i].Should().Be(0f);
    }
}
