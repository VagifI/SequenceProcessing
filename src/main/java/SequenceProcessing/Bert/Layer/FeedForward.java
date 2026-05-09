package SequenceProcessing.Bert.Layer;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class FeedForward {
    private final double[][] W1;
    private final double[][] W2;
    private final double[] b1;
    private final double[] b2;

    public FeedForward(BertConfig config) {
        this.W1 = MathUtils.randomMatrix(config.hiddenSize, config.intermediateSize);
        this.b1 = MathUtils.zeros(config.intermediateSize);

        this.W2 = MathUtils.randomMatrix(config.intermediateSize, config.hiddenSize);
        this.b2 = MathUtils.zeros(config.hiddenSize);
    }

    public double[][] forward(double[][] x) {
        double[][] hidden = MathUtils.addBias(MathUtils.multiply(x, W1), b1);
        hidden = MathUtils.gelu(hidden);

        return MathUtils.addBias(MathUtils.multiply(hidden, W2), b2);
    }
}