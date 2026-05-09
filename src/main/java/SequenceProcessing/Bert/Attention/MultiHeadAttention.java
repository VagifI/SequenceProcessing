package SequenceProcessing.Bert.Attention;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class MultiHeadAttention {
    private final int numHeads;
    private final int headSize;
    private final int hiddenSize;
    private final ScaledDotProductAttention scaledDotProductAttention;

    private final double[][] Wq;
    private final double[][] Wk;
    private final double[][] Wv;
    private final double[][] Wo;

    private final double[] bq, bk, bv, bo;

    public MultiHeadAttention(BertConfig config) {
        this.numHeads = config.numAttentionHeads;
        this.hiddenSize = config.hiddenSize;
        this.headSize = config.hiddenSize / config.numAttentionHeads;
        this.scaledDotProductAttention = new ScaledDotProductAttention();

        this.Wq = MathUtils.randomMatrix(hiddenSize, hiddenSize);
        this.Wk = MathUtils.randomMatrix(hiddenSize, hiddenSize);
        this.Wv = MathUtils.randomMatrix(hiddenSize, hiddenSize);
        this.Wo = MathUtils.randomMatrix(hiddenSize, hiddenSize);

        this.bq = MathUtils.zeros(hiddenSize);
        this.bk = MathUtils.zeros(hiddenSize);
        this.bv = MathUtils.zeros(hiddenSize);
        this.bo = MathUtils.zeros(hiddenSize);
    }

    public double[][] forward(double[][] inputSequence, double[][] mask) {
        int seqLength = inputSequence.length;

        double[][] Q = MathUtils.addBias(MathUtils.multiply(inputSequence, Wq), bq);
        double[][] K = MathUtils.addBias(MathUtils.multiply(inputSequence, Wk), bk);
        double[][] V = MathUtils.addBias(MathUtils.multiply(inputSequence, Wv), bv);

        double[][] multiHeadOutput = new double[seqLength][hiddenSize];

        for (int h = 0; h < numHeads; h++) {
            double[][] qHead = new double[seqLength][headSize];
            double[][] kHead = new double[seqLength][headSize];
            double[][] vHead = new double[seqLength][headSize];

            int startCol = h * headSize;
            for (int i = 0; i < seqLength; i++) {
                System.arraycopy(Q[i], startCol, qHead[i], 0, headSize);
                System.arraycopy(K[i], startCol, kHead[i], 0, headSize);
                System.arraycopy(V[i], startCol, vHead[i], 0, headSize);
            }

            double[][] headOutput = scaledDotProductAttention.forward(qHead, kHead, vHead, mask);

            for (int i = 0; i < seqLength; i++) {
                System.arraycopy(headOutput[i], 0, multiHeadOutput[i], startCol, headSize);
            }
        }

        return MathUtils.addBias(MathUtils.multiply(multiHeadOutput, Wo), bo);
    }
}