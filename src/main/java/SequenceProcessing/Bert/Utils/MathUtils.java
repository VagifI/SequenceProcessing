package SequenceProcessing.Bert.Utils;

import java.util.Random;

public class MathUtils {
    private static final Random random = new Random(42);

    public static double[][] multiply(double[][] a, double[][] b) {
        int r1 = a.length, c1 = a[0].length, c2 = b[0].length;
        double[][] result = new double[r1][c2];
        for (int i = 0; i < r1; i++) {
            for (int k = 0; k < c1; k++) {
                double temp = a[i][k];
                for (int j = 0; j < c2; j++) {
                    result[i][j] += temp * b[k][j];
                }
            }
        }
        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {
        int rows = a.length, cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    public static double[][] addBias(double[][] matrix, double[] bias) {
        int rows = matrix.length, cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] + bias[j];
            }
        }
        return result;
    }

    public static double[][] gelu(double[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = matrix[i][j];

                result[i][j] = 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
            }
        }
        return result;
    }

    public static double[][] randomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextGaussian() * 0.02;
            }
        }
        return matrix;
    }

    public static double[] zeros(int size) {
        return new double[size];
    }
}