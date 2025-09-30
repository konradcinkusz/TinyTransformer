namespace TinyTransformer.Tests.LayersTests;
public class EmbeddingTests : TestsBase
{

    [Fact]
    public void Lookup_ReturnsCorrectRowForSingleToken()
    {
        int vocab = 5;
        int dModel = 3;
        int seed = 123;
        var rnd = new Random(seed);
        var emd = new Embedding(vocab, dModel, rnd);

        //prepare token
        var tokens = new[] { 2 };

        //Act 
        var result = emd.Lookup(tokens);

        //Assert
        result.GetLength(0).Should().Be(1); // one token -> one row
        result.GetLength(1).Should().Be(dModel); // embedding size
    }

    [Fact]
    public void Lookup_ThrowsForOutOfRangeToken()
    {
        int vocab = 5;
        int dModel = 3;
        int seed = 123;
        var rnd = new Random(seed);
        var emd = new Embedding(vocab, dModel, rnd);

        //Act and Assert
        Action action = () => emd.Lookup([7]); //7 > vocab - 1
        action.Should().Throw<ArgumentException>();
    }

}
