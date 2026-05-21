package SequenceProcessing.Classification;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import ComputationalGraph.Function.*;
import ComputationalGraph.Node.*;
import Math.Tensor;
import SequenceProcessing.Functions.*;
import SequenceProcessing.Parameters.BertParameter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class BertModel extends ComputationalGraph implements Serializable {

    public BertModel(BertParameter parameter) {
        super(parameter);
    }

    private ComputationalNode layerNormalization(ComputationalNode input, BertParameter parameter, int[] lnIndex) {
        ArrayList<Double> data = new ArrayList<>();

        ComputationalNode inputMean = this.addEdge(input, new Mean());
        ComputationalNode meanMinus = this.addEdge(inputMean, new Negation());
        ComputationalNode inputMinusMean = this.addAdditionEdge(input, meanMinus, false);

        ComputationalNode variance = this.addEdge(inputMinusMean, new Variance());
        ComputationalNode rootVariance = this.addEdge(variance, new SquareRoot(parameter.getEpsilon()));
        ComputationalNode inverseRootVariance = this.addEdge(rootVariance, new Inverse());
        ComputationalNode normalized = this.addEdge(inputMinusMean, inverseRootVariance, false, true);

        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getGammaValue(lnIndex[0]));
            lnIndex[0]++;
        }
        ComputationalNode gammaInput = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}), true);
        ComputationalNode normGamma = this.addEdge(normalized, gammaInput);

        data.clear();
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getBetaValue(lnIndex[1]));
            lnIndex[1]++;
        }
        ComputationalNode betaInput = new ComputationalNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}));

        return this.addAdditionEdge(normGamma, betaInput, false);
    }

    private ComputationalNode multiHeadAttention(ComputationalNode input, BertParameter parameter, Random random) {
        ArrayList<ComputationalNode> heads = new ArrayList<>();
        for (int i = 0; i < parameter.getN(); i++) {
            ComputationalNode wq = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode wk = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode wv = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));

            ComputationalNode q = this.addEdge(input, wq);
            ComputationalNode k = this.addEdge(input, wk);
            ComputationalNode v = this.addEdge(input, wv);

            ComputationalNode kTranspose = this.addEdge(k, new Transpose());
            ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
            ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / Math.sqrt(parameter.getDk())));
            ComputationalNode sQkDk = this.addEdge(qkDk, new Softmax());
            ComputationalNode attention = this.addEdge(sQkDk, v);
            heads.add(attention);
        }
        ConcatenatedNode concatenated = (ConcatenatedNode) this.concatEdges(heads, 1);
        ComputationalNode wo = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
        return this.addEdge(concatenated, wo);
    }

    private ComputationalNode feedForward(ComputationalNode input, BertParameter parameter, Random random) {
        int intermediateSize = parameter.getL() * 4;
        ComputationalNode w1 = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), intermediateSize, random), new int[]{parameter.getL(), intermediateSize}));
        ComputationalNode hidden = this.addEdge(input, w1);

        ComputationalNode activated = this.addEdge(hidden, parameter.getActivationFunction());

        ComputationalNode w2 = new MultiplicationNode(new Tensor(parameter.initializeWeights(intermediateSize, parameter.getL(), random), new int[]{intermediateSize, parameter.getL()}));
        return this.addEdge(activated, w2);
    }

    @Override
    public void train(ArrayList<Tensor> trainSet) {
        BertParameter parameter = (BertParameter) this.parameters;
        int[] lnIndex = new int[2];
        Random random = new Random(parameter.getSeed());

        ComputationalNode inputNode = new ComputationalNode();
        this.inputNodes.add(inputNode);

        ComputationalNode currentContext = inputNode;

        for (int layer = 0; layer < parameter.getNumLayers(); layer++) {
            ComputationalNode attentionOut = multiHeadAttention(currentContext, parameter, random);
            ComputationalNode add1 = this.addAdditionEdge(currentContext, attentionOut, false);
            ComputationalNode norm1 = layerNormalization(add1, parameter, lnIndex);

            ComputationalNode ffnOut = feedForward(norm1, parameter, random);
            ComputationalNode add2 = this.addAdditionEdge(norm1, ffnOut, false);
            currentContext = layerNormalization(add2, parameter, lnIndex);
        }


        ComputationalNode mlmWeights = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
        ComputationalNode mlmLogits = this.addEdge(currentContext, mlmWeights);
        this.outputNode = this.addEdge(mlmLogits, new Softmax());

        ComputationalNode classLabelNode = new ComputationalNode();
        this.inputNodes.add(classLabelNode);
        this.addLoss(classLabelNode);

        System.out.println("Начинаем обучение модели (Эпох: " + parameter.getEpoch() + ")...");

        for (int i = 0; i < parameter.getEpoch(); i++) {
            this.shuffle(trainSet, random);
            double totalEpochLoss = 0.0;

            for (Tensor instance : trainSet) {
                this.inputNodes.get(0).setValue(instance);
                this.inputNodes.get(1).setValue(instance);

                this.forwardCalculation();


                this.backpropagation();
            }

            System.out.println("Эпоха [" + (i + 1) + "/" + parameter.getEpoch() + "] успешно завершена!");
        }
    }

    @Override
    protected ArrayList<Double> getOutputValue() {
        ArrayList<Double> classLabels = new ArrayList<>();
        if (this.outputNode == null || this.outputNode.getValue() == null) {
            return classLabels;
        }
        Tensor value = this.outputNode.getValue();
        for (int i = 0; i < value.getShape()[0]; i++) {
            double max = -Double.MAX_VALUE;
            double index = -1.0;
            for (int j = 0; j < value.getShape()[1]; j++) {
                double currentValue = value.getValue(new int[]{i, j});
                if (currentValue > max) {
                    max = currentValue;
                    index = j;
                }
            }
            classLabels.add(index);
        }
        return classLabels;
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int correctCount = 0;
        int totalCount = 0;

        for (Tensor instance : testSet) {
            ArrayList<Double> predictedLabels = this.predict();
            if (predictedLabels != null) {
                totalCount += predictedLabels.size();
            }
        }

        double accuracy = totalCount > 0 ? (correctCount + 0.0) / totalCount : 0.0;
        return new ClassificationPerformance(accuracy);
    }
}