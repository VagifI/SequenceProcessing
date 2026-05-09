package SequenceProcessing.Bert.Training;

import SequenceProcessing.Bert.Heads.MaskedLMHead;
import SequenceProcessing.Bert.Heads.NextSentencePredictionHead;
import SequenceProcessing.Bert.Model.BertConfig;
import SequenceProcessing.Bert.Model.BertModel;

public class BertTrainer {
    private final BertModel model;
    private final MaskedLMHead mlmHead;
    private final NextSentencePredictionHead nspHead;
    private final Masker masker;

    public BertTrainer(BertConfig config, BertModel model) {
        this.model = model;
        this.mlmHead = new MaskedLMHead(config);
        this.nspHead = new NextSentencePredictionHead(config);
        this.masker = new Masker(config);
    }

    public double trainStep(int[] originalTokens, int[] segmentIds, double[][] mask, int nspLabel) {
        Masker.MaskedResult maskedData = masker.applyMasking(originalTokens);

        double[][] contextualEmbeddings = model.forward(maskedData.maskedTokens, segmentIds, mask);

        double[][] mlmLogits = mlmHead.forward(contextualEmbeddings);
        double[] nspLogits = nspHead.forward(contextualEmbeddings);

        double mlmLoss = Loss.computeMLMLoss(mlmLogits, maskedData.labels);
        double nspLoss = Loss.computeNSPLoss(nspLogits, nspLabel);
        double totalLoss = mlmLoss + nspLoss;

        return totalLoss;
    }
}