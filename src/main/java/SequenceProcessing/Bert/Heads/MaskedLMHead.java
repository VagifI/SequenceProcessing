package SequenceProcessing.Bert.Heads;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class MaskedLMHead {
    private final double[][] weights;
    private final double[] bias;      // [vocabSize]

    public MaskedLMHead(BertConfig config) {
        this.weights = MathUtils.randomMatrix(config.hiddenSize, config.vocabSize);
        this.bias = MathUtils.zeros(config.vocabSize);
    }

    public double[][] forward(double[][] hiddenStates) {

        return MathUtils.addBias(MathUtils.multiply(hiddenStates, weights), bias);
    }
}