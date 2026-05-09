package SequenceProcessing.Parameters;

import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.NeuralNetworkParameter;
import ComputationalGraph.Function.Function;

import java.io.Serializable;
import java.util.ArrayList;

public class BertParameter extends NeuralNetworkParameter implements Serializable {
    private final int L; // Hidden size
    private final int N; // Количество голов внимания
    private final int numLayers; // Количество слоев энкодера
    private final int V; // Размер словаря
    private final double epsilon;
    private final Function activationFunction;
    private final ArrayList<Double> gammaValues;
    private final ArrayList<Double> betaValues;

    public BertParameter(int seed, int epoch, ComputationalGraph.Optimizer.Optimizer optimizer,
                         Initialization initialization, ComputationalGraph.Loss.Loss loss,
                         int hiddenSize, int numAttentionHeads, int numLayers, int vocabularySize, double epsilon,
                         Function activationFunction, ArrayList<Double> gammaValues, ArrayList<Double> betaValues) {
        super(seed, epoch, optimizer, initialization, loss, 0.0, -1);
        this.L = hiddenSize;
        this.N = numAttentionHeads;
        this.numLayers = numLayers;
        this.V = vocabularySize;
        this.epsilon = epsilon;
        this.activationFunction = activationFunction;
        this.gammaValues = gammaValues;
        this.betaValues = betaValues;
    }

    public int getL() { return L; }
    public int getN() { return N; }
    public int getDk() { return L / N; }
    public int getNumLayers() { return numLayers; }
    public int getV() { return V; }
    public double getEpsilon() { return epsilon; }
    public Function getActivationFunction() { return activationFunction; }
    public double getGammaValue(int index) { return gammaValues.get(index); }
    public double getBetaValue(int index) { return betaValues.get(index); }
}