package SequenceProcessing.Classification;

import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss; // Убедись, что импорт совпадает с тем, что сработало у тебя
import ComputationalGraph.Optimizer.Adam;
import ComputationalGraph.Function.Tanh;
import SequenceProcessing.Bert.Tokenizer.HybridTokenizer;
import SequenceProcessing.Bert.Tokenizer.Vocabulary;
import SequenceProcessing.Parameters.BertParameter;
import Util.FileUtils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class BertMain {

    public static void main(String[] args) {
        System.out.println("=== ИНИЦИАЛИЗАЦИЯ ПРОЕКТА BERT ===");

        // 1. Настраиваем гиперпараметры
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
                new CrossEntropyLoss(), // Твоя функция потерь
                hiddenSize, numHeads, numLayers, vocabSize, epsilon,
                new Tanh(),
                gammaValues, betaValues
        );

        System.out.println("2. Сборка Вычислительного Графа...");
        BertModel bert = new BertModel(parameter);
        System.out.println("   [OK] BERT успешно собран! Слоев: " + parameter.getNumLayers());

        System.out.println("\n=== ТЕСТИРОВАНИЕ ТОКЕНИЗАТОРА НА ДАННЫХ ПРОФЕССОРА ===");
        try {
            // Загружаем словарь
            Vocabulary vocab = new Vocabulary("vocab.txt");
            HybridTokenizer tokenizer = new HybridTokenizer(vocab);

            // Выбираем файл из папки resources (можешь поменять на atis-en.txt или любой другой)
            String fileName = "atis-tr.txt";
            System.out.println("Читаем датасет: " + fileName + "\n");

            InputStreamReader fr = new InputStreamReader(FileUtils.getInputStream(fileName));
            BufferedReader br = new BufferedReader(fr);

            String line;
            int count = 0;

            // Читаем первые 5 строк текста для проверки
            while ((line = br.readLine()) != null && count < 5) {
                if (line.trim().isEmpty() || line.contains("<S>") || line.contains("</S>")) {
                    continue; // Пропускаем пустые строки и теги начала/конца
                }

                // Очищаем строку от лишних пробелов (убираем метки, если они есть)
                String sentence = line.trim();

                // Прогоняем предложение через наш гибридный алгоритм
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