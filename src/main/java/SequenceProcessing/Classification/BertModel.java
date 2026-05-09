package SequenceProcessing.Classification;

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

    // LayerNorm теперь использует единый массив из BertParameter
    private ComputationalNode layerNormalization(ComputationalNode input, BertParameter parameter, int[] lnIndex) {
        ArrayList<Double> data = new ArrayList<>();

        // (X - Mean)
        ComputationalNode inputMean = this.addEdge(input, new Mean());
        ComputationalNode meanMinus = this.addEdge(inputMean, new Negation());
        ComputationalNode inputMinusMean = this.addAdditionEdge(input, meanMinus, false);

        // Variance -> sqrt -> Inverse
        ComputationalNode variance = this.addEdge(inputMinusMean, new Variance());
        ComputationalNode rootVariance = this.addEdge(variance, new SquareRoot(parameter.getEpsilon()));
        ComputationalNode inverseRootVariance = this.addEdge(rootVariance, new Inverse());
        ComputationalNode normalized = this.addEdge(inputMinusMean, inverseRootVariance, false, true);

        // Умножение на Gamma
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getGammaValue(lnIndex[0]));
            lnIndex[0]++;
        }
        ComputationalNode gammaInput = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}), true);
        ComputationalNode normGamma = this.addEdge(normalized, gammaInput);

        // Сложение с Beta
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

        // Используем функцию активации из параметров
        ComputationalNode activated = this.addEdge(hidden, parameter.getActivationFunction());

        ComputationalNode w2 = new MultiplicationNode(new Tensor(parameter.initializeWeights(intermediateSize, parameter.getL(), random), new int[]{intermediateSize, parameter.getL()}));
        return this.addEdge(activated, w2);
    }

    @Override
    public void train(ArrayList<Tensor> trainSet) {
        BertParameter parameter = (BertParameter) this.parameters;
        int[] lnIndex = new int[2]; // lnIndex[0] для Gamma, lnIndex[1] для Beta
        Random random = new Random(parameter.getSeed());

        // Входные данные (эмбеддинги)
        ComputationalNode inputNode = new MultiplicationNode(false, true);
        this.inputNodes.add(inputNode);
        ComputationalNode currentContext = inputNode;

        // Построение N слоев энкодера
        for (int layer = 0; layer < parameter.getNumLayers(); layer++) {
            ComputationalNode attentionOut = multiHeadAttention(currentContext, parameter, random);
            ComputationalNode add1 = this.addAdditionEdge(currentContext, attentionOut, false);
            ComputationalNode norm1 = layerNormalization(add1, parameter, lnIndex);

            ComputationalNode ffnOut = feedForward(norm1, parameter, random);
            ComputationalNode add2 = this.addAdditionEdge(norm1, ffnOut, false);
            currentContext = layerNormalization(add2, parameter, lnIndex);
        }

        // Выходной слой MLM
        ComputationalNode mlmWeights = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getV(), random), new int[]{parameter.getL(), parameter.getV()}));
        ComputationalNode mlmLogits = this.addEdge(currentContext, mlmWeights);
        this.outputNode = this.addEdge(mlmLogits, new Softmax());

        // Настройка функции потерь
        ComputationalNode classLabelNode = new ComputationalNode();
        this.inputNodes.add(classLabelNode);
        this.addLoss(classLabelNode);

        // Цикл обучения с использованием Tensor датасетов
        for (int i = 0; i < parameter.getEpoch(); i++) {
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                // Прямой и обратный проход.
                // В реальном коде здесь должна быть обвязка для заполнения inputNode и classLabelNode
                // из тензора 'instance', аналогично методу createInputTensors у профессора.
                this.forwardCalculation();
                this.backpropagation();
            }
            parameter.getOptimizer().setLearningRate();
        }
    }
}