namespace TinyTransformer.Core.Interfaces;

//neural net layer
public interface ILayer
{
    /// <summary>
    /// Forward pass through the layer
    /// Input and output are tensors
    /// 
    /// TODO: add Tensor class
    /// </summary>
    /// <param name="X"></param>
    /// <returns></returns>
    float[,] Forward(float[,] X);
}
