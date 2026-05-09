package SequenceProcessing.Bert.Embedding;

import SequenceProcessing.Bert.Layer.LayerNorm;
import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Utils.MathUtils;

public class BertEmbedding {
    private final TokenEmbedding tokenEmbedding;
    private final PositionEmbedding positionEmbedding;
    private final SegmentEmbedding segmentEmbedding;
    private final LayerNorm layerNorm;

    public BertEmbedding(BertConfig config) {
        this.tokenEmbedding = new TokenEmbedding(config);
        this.positionEmbedding = new PositionEmbedding(config);
        this.segmentEmbedding = new SegmentEmbedding(config);
        this.layerNorm = new LayerNorm(config);
    }

    public double[][] forward(int[] tokenIds, int[] segmentIds) {
        int seqLength = tokenIds.length;

        double[][] tokEmb = tokenEmbedding.forward(tokenIds);
        double[][] posEmb = positionEmbedding.forward(seqLength);
        double[][] segEmb = segmentEmbedding.forward(segmentIds);

        double[][] embeddings = new double[seqLength][tokEmb[0].length];
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < tokEmb[0].length; j++) {
                embeddings[i][j] = tokEmb[i][j] + posEmb[i][j] + segEmb[i][j];
            }
        }

        return layerNorm.forward(embeddings);
    }
}