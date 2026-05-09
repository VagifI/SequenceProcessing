package SequenceProcessing.Bert.Model;

public class BertConfig {
    public final int vocabSize;
    public final int hiddenSize;
    public final int numHiddenLayers;
    public final int numAttentionHeads;
    public final int intermediateSize;
    public final int maxPositionEmbeddings;
    public final double dropoutProb;
    public final double layerNormEps;

    public BertConfig(int vocabSize, int hiddenSize, int numHiddenLayers,
                      int numAttentionHeads, int intermediateSize,
                      int maxPositionEmbeddings, double dropoutProb, double layerNormEps) {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.numHiddenLayers = numHiddenLayers;
        this.numAttentionHeads = numAttentionHeads;
        this.intermediateSize = intermediateSize;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.dropoutProb = dropoutProb;
        this.layerNormEps = layerNormEps;
    }

    // Дефолтная конфигурация для BERT-base
    public static BertConfig bertBase() {
        return new BertConfig(30522, 768, 12, 12, 3072, 512, 0.1, 1e-12);
    }
}