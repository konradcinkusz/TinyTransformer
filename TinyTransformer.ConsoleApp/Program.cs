using TinyTransformer.Core.Layers;

namespace TinyTransformer.ConsoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int vocabSize = 20; //token IDs range from 0..19
            int dModel = 16; //vector size per token
            int dK = 16; //attention head size
            int ffHidden = 32; //feed forward inner layer size
            int seqLen = 5; //number of tokens in the input sequence

            var rnd = new Random(0);

            int[] tokens = [3, 7, 7, 2, 9];

            //Components
            var embedding = new Embedding(vocabSize, dModel, rnd);
            var posEncoding = 1;//
            var block = new TransformerEncoderBlock(dModel, dK, ffHidden, rnd);

            //Forward 
            float[,] X = embedding.Lookup(tokens);
            //positional Encoding build
            var encoded = block.Forward(X);

            //Inspect
            Console.WriteLine("Encoded sequence");

        }
    }
}
