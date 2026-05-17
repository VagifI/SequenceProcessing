package SequenceProcessing.Bert.Tokenizer;

import MorphologicalAnalysis.FsmMorphologicalAnalyzer;
import MorphologicalAnalysis.FsmParse;
import MorphologicalAnalysis.FsmParseList;

import java.io.*;
import java.util.*;

public class VocabBuilder {

    public static void build(String[] fileNames, String outputFilePath) {
        System.out.println("Начинаем обучение словаря...");
        HashMap<String, Integer> tokenFrequencies = new HashMap<>();
        FsmMorphologicalAnalyzer analyzer;

        try {
            analyzer = new FsmMorphologicalAnalyzer();
        } catch (Exception e) {
            System.out.println("Ошибка загрузки анализатора: " + e.getMessage());
            return;
        }

        for (String fileName : fileNames) {
            System.out.println("Обработка файла: " + fileName);
            try (InputStream is = Util.FileUtils.getInputStream(fileName);
                 BufferedReader br = new BufferedReader(new InputStreamReader(is))) {

                String line;
                while ((line = br.readLine()) != null) {
                    if (line.trim().isEmpty() || line.contains("<S>") || line.contains("</S>")) continue;

                    String[] words = line.trim().split("\\s+");
                    for (String word : words) {
                        List<String> tokens = extractMorphemes(word, analyzer);
                        for (String token : tokens) {
                            tokenFrequencies.put(token, tokenFrequencies.getOrDefault(token, 0) + 1);
                        }
                    }
                }
            } catch (Exception e) {
                System.out.println("Не удалось прочитать файл " + fileName + ": " + e.getMessage());
            }
        }

        List<Map.Entry<String, Integer>> sortedTokens = new ArrayList<>(tokenFrequencies.entrySet());
        sortedTokens.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFilePath))) {
            writer.println("[PAD]");
            writer.println("[UNK]");
            writer.println("[CLS]");
            writer.println("[SEP]");
            writer.println("[MASK]");

            int count = 5;
            for (Map.Entry<String, Integer> entry : sortedTokens) {
                writer.println(entry.getKey());
                count++;

                if (count >= 30522) break;
            }
            System.out.println("Словарь успешно создан! Сохранено уникальных токенов: " + count);
        } catch (IOException e) {
            System.out.println("Ошибка при записи словаря: " + e.getMessage());
        }
    }

    private static List<String> extractMorphemes(String word, FsmMorphologicalAnalyzer analyzer) {
        List<String> tokens = new ArrayList<>();
        try {
            FsmParseList parses = analyzer.morphologicalAnalysis(word);
            if (parses.size() > 0) {
                FsmParse bestParse = parses.getFsmParse(0);
                String transition = bestParse.transitionList();
                if (transition != null && !transition.isEmpty()) {
                    String[] tags = transition.split("\\+");
                    tokens.add(tags[0]);
                    for (int i = 1; i < tags.length; i++) {
                        tokens.add("##" + tags[i].toLowerCase());
                    }
                    return tokens;
                }
            }
        } catch (Exception ignored) {}

        tokens.add(word);
        return tokens;
    }
}