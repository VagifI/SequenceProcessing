package SequenceProcessing.Bert.Layer;

import SequenceProcessing.Bert.Model.BertConfig;
import java.util.Arrays;

public class LayerNorm {
    private final double epsilon;
    private final double[] gamma;
    private final double[] beta;

    public LayerNorm(BertConfig config) {
        this.epsilon = config.layerNormEps;
        this.gamma = new double[config.hiddenSize];
        this.beta = new double[config.hiddenSize];

        Arrays.fill(this.gamma, 1.0);
        Arrays.fill(this.beta, 0.0);
    }

    public double[][] forward(double[][] x) {
        int seqLength = x.length;
        int hiddenSize = x[0].length;
        double[][] out = new double[seqLength][hiddenSize];

        for (int i = 0; i < seqLength; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += x[i][j];
            }
            double mean = sum / hiddenSize;

            double varianceSum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                varianceSum += Math.pow(x[i][j] - mean, 2);
            }
            double variance = varianceSum / hiddenSize;

            for (int j = 0; j < hiddenSize; j++) {
                double normalized = (x[i][j] - mean) / Math.sqrt(variance + epsilon);
                out[i][j] = normalized * gamma[j] + beta[j];
            }
        }
        return out;
    }
}