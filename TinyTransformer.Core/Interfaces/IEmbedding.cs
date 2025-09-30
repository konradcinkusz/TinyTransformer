namespace TinyTransformer.Core.Interfaces;

public interface IEmbedding
{
    float[,] Lookup(int[] tokenIds);
}