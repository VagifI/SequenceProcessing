package SequenceProcessing.Bert.Heads;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class NextSentencePredictionHead {
    private final double[][] weights; // [hiddenSize, 2]
    private final double[] bias;      // [2]

    public NextSentencePredictionHead(BertConfig config) {
        this.weights = MathUtils.randomMatrix(config.hiddenSize, 2);
        this.bias = MathUtils.zeros(2);
    }

    public double[] forward(double[][] hiddenStates) {
        double[][] clsTokenEmbedding = new double[1][hiddenStates[0].length];
        clsTokenEmbedding[0] = hiddenStates[0];

        double[][] logitsMatrix = MathUtils.addBias(MathUtils.multiply(clsTokenEmbedding, weights), bias);

        return logitsMatrix[0];
    }
}