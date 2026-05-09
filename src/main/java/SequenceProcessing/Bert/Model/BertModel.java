package SequenceProcessing.Bert.Model;

import SequenceProcessing.Bert.Embedding.BertEmbedding;
import SequenceProcessing.Bert.Layer.EncoderBlock;

public class BertModel {
    private final BertConfig config;
    private final BertEmbedding embeddings;
    private final EncoderBlock[] encoderLayers;

    public BertModel(BertConfig config) {
        this.config = config;
        this.embeddings = new BertEmbedding(config);

        this.encoderLayers = new EncoderBlock[config.numHiddenLayers];
        for (int i = 0; i < config.numHiddenLayers; i++) {
            encoderLayers[i] = new EncoderBlock(config);
        }
    }

    public double[][] forward(int[] tokenIds, int[] segmentIds, double[][] mask) {
        double[][] hiddenStates = embeddings.forward(tokenIds, segmentIds);

        for (int i = 0; i < config.numHiddenLayers; i++) {
            hiddenStates = encoderLayers[i].forward(hiddenStates, mask);
        }

        return hiddenStates;
    }
}