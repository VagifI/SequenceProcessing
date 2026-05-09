package SequenceProcessing.Bert.Embedding;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class PositionEmbedding {
    private final double[][] weight; // Матрица [maxPositionEmbeddings, hiddenSize]

    public PositionEmbedding(BertConfig config) {
        this.weight = MathUtils.randomMatrix(config.maxPositionEmbeddings, config.hiddenSize);
    }

    public double[][] forward(int seqLength) {
        int hiddenSize = weight[0].length;
        double[][] output = new double[seqLength][hiddenSize];

        for (int i = 0; i < seqLength; i++) {
            System.arraycopy(weight[i], 0, output[i], 0, hiddenSize);
        }
        return output;
    }
}