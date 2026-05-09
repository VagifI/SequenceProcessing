package SequenceProcessing.Bert.Training;

import java.util.Arrays;

public class Loss {
    public static double computeMLMLoss(double[][] logits, int[] targetLabels) {
        double totalLoss = 0.0;
        int count = 0;

        for (int i = 0; i < targetLabels.length; i++) {
            if (targetLabels[i] != -1) {
                double[] tokenLogits = logits[i];
                int correctClass = targetLabels[i];

                totalLoss += crossEntropy(tokenLogits, correctClass);
                count++;
            }
        }
        return count > 0 ? totalLoss / count : 0.0;
    }

    public static double computeNSPLoss(double[] logits, int targetLabel) {
        return crossEntropy(logits, targetLabel);
    }

    private static double crossEntropy(double[] logits, int targetClass) {
        double maxLogit = Arrays.stream(logits).max().orElse(0.0);
        double sumExp = 0.0;

        for (double logit : logits) {
            sumExp += Math.exp(logit - maxLogit);
        }

        // Loss = -log(exp(logit_correct) / sumExp) = -logit_correct + log(sumExp)
        double correctLogit = logits[targetClass] - maxLogit;
        return -correctLogit + Math.log(sumExp);
    }
}