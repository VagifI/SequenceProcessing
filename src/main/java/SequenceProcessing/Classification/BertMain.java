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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class BertMain {

    public static void main(String[] args) {
        System.out.println("=== ИНИЦИАЛИЗАЦИЯ ПРОЕКТА BERT ===");
        int hiddenSize = 768;
        int numHeads = 12;
        int numLayers = 12;
        int vocabSize = 30522;
        double epsilon = 1e-12;

        ArrayList<Double> gammaValues = new ArrayList<>();
        ArrayList<Double> betaValues = new ArrayList<>();
        for (int i = 0; i < numLayers * 2 * hiddenSize; i++) {
            gammaValues.add(1.0);
            betaValues.add(0.0);
        }

        System.out.println("1. Создание параметров модели...");
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

        System.out.println("2. Сборка Вычислительного Графа...");
        BertModel bert = new BertModel(parameter);
        System.out.println("   [OK] BERT успешно собран! Слоев: " + parameter.getNumLayers());
        System.out.println("\n=== ОБУЧЕНИЕ СЛОВАРЯ ===");

        String[] trainingFiles = {
                "atis-tr.txt"
        };

        String newVocabPath = "vocab_real.txt";
        VocabBuilder.build(trainingFiles, newVocabPath);

        System.out.println("\n=== ТЕСТИРОВАНИЕ ТОКЕНИЗАТОРА НА ДАННЫХ ПРОФЕССОРА ===");
        try {
            Vocabulary vocab = new Vocabulary(newVocabPath);
            HybridTokenizer tokenizer = new HybridTokenizer(vocab);

            String fileName = "atis-tr.txt";
            System.out.println("Читаем датасет для проверки: " + fileName + "\n");

            InputStreamReader fr = new InputStreamReader(FileUtils.getInputStream(fileName));
            BufferedReader br = new BufferedReader(fr);

            String line;
            int count = 0;

            while ((line = br.readLine()) != null && count < 5) {
                if (line.trim().isEmpty() || line.contains("<S>") || line.contains("</S>")) {
                    continue;
                }

                String sentence = line.trim();
                ArrayList<Integer> tokenIds = tokenizer.encode(sentence);

                System.out.println("Оригинал : " + sentence);
                System.out.println("Токены ID: " + tokenIds);
                System.out.println("--------------------------------------------------");
                count++;
            }
            br.close();

            System.out.println("\n=== ВСЕ СИСТЕМЫ РАБОТАЮТ ШТАТНО! ===");

        } catch (Exception e) {
            System.out.println("Ошибка при чтении файла или словаря: " + e.getMessage());
            e.printStackTrace();
        }
    }
}