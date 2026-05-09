package SequenceProcessing.Bert.Attention;

import java.util.Arrays;

public class ScaledDotProductAttention {
    public double[][] forward(double[][] query, double[][] key, double[][] value, double[][] mask) {
        int seqLength = query.length;
        int dK = query[0].length;

        double[][] scores = multiplyMatrices(query, transpose(key));
        double scale = Math.sqrt(dK);

        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < seqLength; j++) {
                scores[i][j] /= scale;

                if (mask != null && mask[i][j] == 0.0) {
                    scores[i][j] = Double.NEGATIVE_INFINITY; // Чтобы softmax дал 0
                }
            }
        }

        double[][] attentionWeights = applySoftmax(scores);

        return multiplyMatrices(attentionWeights, value);
    }

    private double[][] applySoftmax(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            double max = Arrays.stream(matrix[i]).max().orElse(0.0);
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.exp(matrix[i][j] - max); // Вычитание max для численной стабильности
                sum += result[i][j];
            }
            for (int j = 0; j < cols; j++) {
                result[i][j] /= sum;
            }
        }
        return result;
    }

    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    private double[][] multiplyMatrices(double[][] a, double[][] b) {
        int r1 = a.length;
        int c1 = a[0].length;
        int c2 = b[0].length;
        double[][] result = new double[r1][c2];
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }
}