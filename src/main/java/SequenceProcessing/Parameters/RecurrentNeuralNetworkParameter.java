package SequenceProcessing.Parameters;

import ComputationalGraph.Function;
import ComputationalGraph.Initialization;
import ComputationalGraph.NeuralNetworkParameter;
import ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.ArrayList;

public class RecurrentNeuralNetworkParameter extends NeuralNetworkParameter implements Serializable {

    private final ArrayList<Integer> hiddenLayers;
    private final ArrayList<Function> functions;
    private final int classLabelSize;

    public RecurrentNeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Initialization initialization, ArrayList<Integer> hiddenLayers, ArrayList<Function> functions, int classLabelSize) {
        super(seed, epoch, optimizer, initialization);
        this.hiddenLayers = hiddenLayers;
        this.functions = functions;
        this.classLabelSize = classLabelSize;
    }

    public int size() {
        return hiddenLayers.size();
    }

    public int getClassLabelSize() {
        return classLabelSize;
    }

    public Function getActivationFunction(int index) {
        return functions.get(index);
    }

    public Integer getHiddenLayer(int index) {
        return hiddenLayers.get(index);
    }
}
