package SequenceProcessing.Classification;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.ConcatenatedNode;
import ComputationalGraph.Node.MultiplicationNode;
import Math.*;
import SequenceProcessing.Functions.RemoveBias;
import SequenceProcessing.Functions.Switch;
import SequenceProcessing.Parameters.*;

public class RecurrentNeuralNetworkModel extends ComputationalGraph implements Serializable {

    protected final int wordEmbeddingLength;
    protected ArrayList<Switch> switches;

    public RecurrentNeuralNetworkModel(NeuralNetworkParameter parameter, int wordEmbeddingLength) {
        super(parameter);
        this.wordEmbeddingLength = wordEmbeddingLength;
        this.switches = new ArrayList<>();
    }

    protected void createInputTensors(Tensor instance, int outputSize) {
        ArrayList<Integer> classLabels = new ArrayList<>();
        int timeStep = (instance.getShape()[0] / (wordEmbeddingLength + 1));
        int j = 0;
        for (int i = 0; i < this.inputNodes.size() - 1; i++) {
            if (i < timeStep) {
                this.switches.get(i).setTurn(true);
                ArrayList<Double> values = new ArrayList<>();
                for (int k = 0; k < wordEmbeddingLength; k++) {
                    values.add(instance.getValue(new int[]{j}));
                    j++;
                }
                classLabels.add((int) instance.getValue(new int[]{j}));
                j++;
                inputNodes.get(i).setValue(new Tensor(values, new int[]{1, values.size()}));
            } else {
                this.switches.get(i).setTurn(false);
                ArrayList<Double> values = new ArrayList<>();
                for (int k = 0; k < wordEmbeddingLength; k++) {
                    values.add(0.0);
                    j++;
                }
                classLabels.add(0);
                j++;
                inputNodes.get(i).setValue(new Tensor(values, new int[]{1, values.size()}));
            }
        }
        ArrayList<Double> input3Values = new ArrayList<>();
        for (Integer classLabel : classLabels) {
            for (int in = 0; in < outputSize; in++) {
                if (in == classLabel) {
                    input3Values.add(1.0);
                } else {
                    input3Values.add(0.0);
                }
            }
        }
        inputNodes.get(this.inputNodes.size() - 1).setValue(new Tensor(input3Values, new int[]{classLabels.size(), outputSize}));
    }

    protected void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters, Random random) {
        for (int i = 0; i < parameters.getEpoch(); i++) {
            System.out.println("Epoch: " + (i + 1));
            // Shuffle
            for (int j = 0; j < trainSet.size(); j++) {
                int i1 = random.nextInt(trainSet.size());
                int i2 = random.nextInt(trainSet.size());
                Tensor tmp = trainSet.get(i1);
                trainSet.set(i1, trainSet.get(i2));
                trainSet.set(i2, tmp);
            }
            for (Tensor instance : trainSet) {
                createInputTensors(instance, ((RecurrentNeuralNetworkParameter) this.getParameters()).getClassLabelSize());
                this.forwardCalculation();
                this.backpropagation(parameters.getOptimizer());
            }
            parameters.getOptimizer().setLearningRate();
        }
    }

    protected int findTimeStep(ArrayList<Tensor> trainSet) {
        int timeStep = -1;
        for (Tensor tensor : trainSet) {
            int size = tensor.getShape()[0];
            if (timeStep < size / (wordEmbeddingLength + 1)) {
                timeStep = size / (wordEmbeddingLength + 1);
            }
        }
        return timeStep;
    }

    // Many-to-Many RNN
    @Override
    public void train(ArrayList<Tensor> trainSet) {
        RecurrentNeuralNetworkParameter parameters = (RecurrentNeuralNetworkParameter) this.getParameters();
        Random random = new Random(parameters.getSeed());
        int timeStep = findTimeStep(trainSet);
        ArrayList<ComputationalNode> weights = new ArrayList<>();
        ArrayList<ComputationalNode> recurrentWeights = new ArrayList<>();
        int currentLength = wordEmbeddingLength + 1;
        for (int i = 0; i < parameters.size(); i++) {
            weights.add(new MultiplicationNode(new Tensor(parameters.initializeWeights(currentLength, parameters.getHiddenLayer(i), random), new int[]{currentLength, parameters.getHiddenLayer(i)})));
            recurrentWeights.add(new MultiplicationNode(new Tensor(parameters.initializeWeights(parameters.getHiddenLayer(i), parameters.getHiddenLayer(i), random), new int[]{parameters.getHiddenLayer(i), parameters.getHiddenLayer(i)})));
            currentLength = parameters.getHiddenLayer(i) + 1;
        }
        weights.add(new MultiplicationNode(new Tensor(parameters.initializeWeights(currentLength, parameters.getClassLabelSize(), random), new int[]{currentLength, parameters.getClassLabelSize()})));
        ArrayList<ComputationalNode> currentOldLayers = new ArrayList<>();
        ArrayList<ComputationalNode> outputNodes = new ArrayList<>();
        for (int k = 0; k < timeStep; k++) {
            this.switches.add(new Switch());
            ArrayList<ComputationalNode> newOldLayers = new ArrayList<>();
            ComputationalNode input = new MultiplicationNode(false, true);
            inputNodes.add(input);
            ComputationalNode current = input;
            for (int i = 0; i < parameters.size(); i++) {
                ComputationalNode aw;
                ComputationalNode aFunction;
                if (!currentOldLayers.isEmpty()) {
                    aw = this.addEdge(current, weights.get(i), false);
                    ComputationalNode oWithoutBias = this.addEdge(currentOldLayers.get(i), new RemoveBias(), false);
                    ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights.get(i), false);
                    ComputationalNode a = this.addAdditionEdge(aw, ou, false);
                    aFunction = this.addEdge(a, parameters.getActivationFunction(i), true);
                } else {
                    aw = this.addEdge(current, weights.get(i), false);
                    aFunction = this.addEdge(aw, parameters.getActivationFunction(i), true);
                }
                current = aFunction;
                newOldLayers.add(aFunction);
            }
            currentOldLayers = newOldLayers;
            ComputationalNode node = this.addEdge(current, weights.get(weights.size() - 1), false);
            outputNodes.add(this.addEdge(node, switches.get(k), false));
        }
        ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(outputNodes, 0);
        this.outputNode = this.addEdge(concatenatedNode, new Softmax(), false);
        ComputationalNode classLabelNode = new ComputationalNode();
        this.inputNodes.add(classLabelNode);
        ArrayList<ComputationalNode> lossInputs = new ArrayList<>();
        lossInputs.add(this.outputNode);
        lossInputs.add(classLabelNode);
        this.addFunctionEdge(lossInputs, parameters.getLossFunction(), false);
        train(trainSet, parameters, random);
    }

    @Override
    protected ArrayList<Double> getOutputValue(ComputationalNode outputNode) {
        ArrayList<Double> classLabels = new ArrayList<>();
        for (int i = 0; i < outputNode.getValue().getShape()[0]; i++) {
            int index = -1;
            double max = Double.MIN_VALUE;
            for (int j = 0; j < outputNode.getValue().getShape()[1]; j++) {
                if (max < outputNode.getValue().getValue(new int[]{i, j})) {
                    max = outputNode.getValue().getValue(new int[]{i, j});
                    index = j;
                }
            }
            classLabels.add((double) index);
        }
        return classLabels;
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            createInputTensors(instance, ((RecurrentNeuralNetworkParameter) this.getParameters()).getClassLabelSize());
            ArrayList<Integer> goldClassLabels = new ArrayList<>();
            ArrayList<Double> classLabelValues = (ArrayList<Double>) this.inputNodes.get(this.inputNodes.size() - 1).getValue().getData();
            for (int i = 0; i < classLabelValues.size(); i++) {
                if (classLabelValues.get(i) == 1.0) {
                    goldClassLabels.add(i % ((RecurrentNeuralNetworkParameter) this.getParameters()).getClassLabelSize());
                }
            }
            ArrayList<Double> classLabels = this.predict();
            for (int j = 0; j < (instance.getShape()[0] / (wordEmbeddingLength + 1)); j++) {
                if (goldClassLabels.get(j).equals(classLabels.get(j).intValue())) {
                    count++;
                }
                total++;
            }
        }
        return new ClassificationPerformance((count + 0.0) / total);
    }
}
