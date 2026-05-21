package SequenceProcessing.Classification;

import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Optimizer.Adam;
import ComputationalGraph.Function.Tanh;
import SequenceProcessing.Bert.Tokenizer.HybridTokenizer;
import SequenceProcessing.Bert.Tokenizer.VocabBuilder;
import SequenceProcessing.Bert.Tokenizer.Vocabulary;
import SequenceProcessing.Parameters.BertParameter;

import Util.FileUtils;
import Math.Tensor;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class BertMain {

    public static void main(String[] args) {
        System.out.println("=== INITIALIZING THE BERT PROJECT ===");

        int hiddenSize = 128;
        int numHeads = 2;
        int numLayers = 2;
        int vocabSize = 30522;
        double epsilon = 1e-12;

        ArrayList<Double> gammaValues = new ArrayList<>();
        ArrayList<Double> betaValues = new ArrayList<>();
        for (int i = 0; i < numLayers * 2 * hiddenSize; i++) {
            gammaValues.add(1.0);
            betaValues.add(0.0);
        }

        System.out.println("1. Creating model parameters...");
        BertParameter parameter = new BertParameter(
                42,
                10,
                new Adam(0.001, 0.0, 0.9, 0.999, epsilon),
                new RandomInitialization(),
                new CrossEntropyLoss(),
                hiddenSize, numHeads, numLayers, vocabSize, epsilon,
                new Tanh(),
                gammaValues, betaValues
        );

        System.out.println("2. Assembling the Computational Graph...");
        BertModel bert = new BertModel(parameter);

        System.out.println("   [OK] BERT successfully assembled! Layers: " + parameter.getNumLayers());
        System.out.println("\n=== VOCABULARY TRAINING ===");

        String[] trainingFiles = {
                "atis-tr.txt",
                "atis-en.txt"
        };
        String newVocabPath = "vocab_real.txt";
        VocabBuilder.build(trainingFiles, newVocabPath);

        Vocabulary vocab = null;
        HybridTokenizer tokenizer = null;
        String fileName = "atis-tr.txt";

        System.out.println("\n=== TESTING THE TOKENIZER ===");
        try {
            vocab = new Vocabulary(newVocabPath);
            tokenizer = new HybridTokenizer(vocab);
            InputStreamReader fr = new InputStreamReader(FileUtils.getInputStream(fileName));
            BufferedReader br = new BufferedReader(fr);
            String line;
            int count = 0;

            while ((line = br.readLine()) != null && count < 5) {
                if (line.trim().isEmpty() || line.contains("<S>") || line.contains("</S>")) continue;
                String sentence = line.trim();
                ArrayList<Integer> tokenIds = tokenizer.encode(sentence);
                System.out.println("Original : " + sentence);
                System.out.println("Tokens ID: " + tokenIds);
                System.out.println("--------------------------------------------------");
                count++;
            }
            br.close();
            System.out.println("\n=== ALL SYSTEMS ARE WORKING ===");

        } catch (Exception e) {
            System.out.println("Error reading file or dictionary: " + e.getMessage());
        }

        if (tokenizer != null) {
            System.out.println("\n=== DATA PREPARATION ===");
            ArrayList<Tensor> trainDataset = new ArrayList<>();

            double[][] embeddings = new double[vocabSize][hiddenSize];
            java.util.Random rand = new java.util.Random(42);
            for(int i = 0; i < vocabSize; i++) {
                for(int j = 0; j < hiddenSize; j++) {
                    embeddings[i][j] = rand.nextGaussian() * 0.01;
                }
            }

            try {
                for (String file : trainingFiles) {
                    System.out.println("Converting tokens from a file: " + file);
                    InputStreamReader frTrain = new InputStreamReader(FileUtils.getInputStream(file));
                    BufferedReader brTrain = new BufferedReader(frTrain);
                    String trainLine;

                    int fileSentenceCount = 0;

                    while ((trainLine = brTrain.readLine()) != null) {
                        if (trainLine.trim().isEmpty() || trainLine.contains("<S>") || trainLine.contains("</S>")) continue;

                        ArrayList<Integer> tokenIds = tokenizer.encode(trainLine.trim());
                        ArrayList<Double> tensorData = new ArrayList<>();

                        for (int id : tokenIds) {
                            int safeId = Math.min(id, vocabSize - 1);
                            for (int j = 0; j < hiddenSize; j++) {
                                tensorData.add(embeddings[safeId][j]);
                            }
                        }

                        Tensor textTensor = new Tensor(tensorData, new int[]{tokenIds.size(), hiddenSize});
                        trainDataset.add(textTensor);

                        fileSentenceCount++;

                        if (fileSentenceCount >= 100) {
                            System.out.println("   [OK] Taken " + fileSentenceCount + " out of " + file);
                            break;
                        }
                    }
                    brTrain.close();
                }

                System.out.println("\nTOTAL: Prepared " + trainDataset.size() + " tensors for training.");

                System.out.println("\n=== START OF TRAINING ===");
                bert.train(trainDataset);

                System.out.println("\n=== TRAINING SUCCESSFULLY COMPLETED ===");

            } catch (Exception e) {
                System.out.println("Error while preparing tensors: " + e.getMessage());
            }

            System.out.println("\n=== SAVING THE MODEL ===");
            String modelPath = "bert_weights.model";

            try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelPath))) {
                oos.writeObject(parameter);
                System.out.println("[OK] Model weights have been successfully saved to file: " + modelPath);

            } catch (java.io.IOException e) {
                System.out.println("Error saving model: " + e.getMessage());
            }
        }
    }
}