package SequenceProcessing.Bert.Embedding;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class TokenEmbedding {
    private final double[][] weight;

    public TokenEmbedding(BertConfig config) {
        this.weight = MathUtils.randomMatrix(config.vocabSize, config.hiddenSize);
    }

    public double[][] forward(int[] tokenIds) {
        int seqLength = tokenIds.length;
        int hiddenSize = weight[0].length;
        double[][] output = new double[seqLength][hiddenSize];

        for (int i = 0; i < seqLength; i++) {
            int tokenId = tokenIds[i];

            System.arraycopy(weight[tokenId], 0, output[i], 0, hiddenSize);
        }
        return output;
    }
}