package SequenceProcessing.Bert.Embedding;

import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class SegmentEmbedding {
    private final double[][] weight; // Матрица [2, hiddenSize], так как сегментов обычно 2 (A и B)

    public SegmentEmbedding(BertConfig config) {
        this.weight = MathUtils.randomMatrix(2, config.hiddenSize);
    }

    public double[][] forward(int[] segmentIds) {
        int seqLength = segmentIds.length;
        int hiddenSize = weight[0].length;
        double[][] output = new double[seqLength][hiddenSize];

        for (int i = 0; i < seqLength; i++) {
            int segmentId = segmentIds[i];
            System.arraycopy(weight[segmentId], 0, output[i], 0, hiddenSize);
        }
        return output;
    }
}