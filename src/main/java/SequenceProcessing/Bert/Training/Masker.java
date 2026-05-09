package SequenceProcessing.Bert.Training;

import SequenceProcessing.Bert.Model.BertConfig;
import java.util.Random;

public class Masker {
    private final int vocabSize;
    private final int maskTokenId = 103;
    private final Random random;

    public Masker(BertConfig config) {
        this.vocabSize = config.vocabSize;
        this.random = new Random(42);
    }

    public static class MaskedResult {
        public final int[] maskedTokens;
        public final int[] labels;

        public MaskedResult(int[] maskedTokens, int[] labels) {
            this.maskedTokens = maskedTokens;
            this.labels = labels;
        }
    }

    public MaskedResult applyMasking(int[] inputTokens) {
        int[] maskedTokens = new int[inputTokens.length];
        int[] labels = new int[inputTokens.length];

        for (int i = 0; i < inputTokens.length; i++) {
            if (inputTokens[i] == 101 || inputTokens[i] == 102) {
                maskedTokens[i] = inputTokens[i];
                labels[i] = -1;
                continue;
            }

            double prob = random.nextDouble();
            if (prob < 0.15) {
                labels[i] = inputTokens[i];

                double maskProb = random.nextDouble();
                if (maskProb < 0.8) {
                    maskedTokens[i] = maskTokenId;
                } else if (maskProb < 0.9) {
                    maskedTokens[i] = random.nextInt(vocabSize);
                } else {
                    maskedTokens[i] = inputTokens[i];
                }
            } else {
                maskedTokens[i] = inputTokens[i];
                labels[i] = -1;
            }
        }
        return new MaskedResult(maskedTokens, labels);
    }
}