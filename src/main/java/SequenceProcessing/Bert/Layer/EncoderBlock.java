package SequenceProcessing.Bert.Layer;

import SequenceProcessing.Bert.Attention.MultiHeadAttention;
import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class EncoderBlock {
    private final MultiHeadAttention attention;
    private final LayerNorm attentionNorm;
    private final FeedForward feedForward;
    private final LayerNorm ffnNorm;

    public EncoderBlock(BertConfig config) {
        this.attention = new MultiHeadAttention(config);
        this.attentionNorm = new LayerNorm(config);
        this.feedForward = new FeedForward(config);
        this.ffnNorm = new LayerNorm(config);
    }

    public double[][] forward(double[][] x, double[][] mask) {
        double[][] attentionOutput = attention.forward(x, mask);
        double[][] add1 = MathUtils.add(x, attentionOutput);
        double[][] norm1 = attentionNorm.forward(add1);

        double[][] ffnOutput = feedForward.forward(norm1);
        double[][] add2 = MathUtils.add(norm1, ffnOutput);
        double[][] output = ffnNorm.forward(add2);

        return output;
    }
}