package SequenceProcessing.Bert.Tokenizer;

import Util.FileUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class Vocabulary implements Serializable {
    private final Map<String, Integer> wordToId;
    private final Map<Integer, String> idToWord;
    public final int unkTokenId;

    public Vocabulary(String vocabFileName) {
        this.wordToId = new HashMap<>();
        this.idToWord = new HashMap<>();
        loadVocab(vocabFileName);
        this.unkTokenId = wordToId.getOrDefault("[UNK]", 100); // Обычно 100 в BERT
    }

    private void loadVocab(String fileName) {
        try {
            InputStreamReader fr = new InputStreamReader(FileUtils.getInputStream(fileName));
            BufferedReader br = new BufferedReader(fr);
            String line;
            int index = 0;
            while ((line = br.readLine()) != null) {
                String token = line.trim();
                wordToId.put(token, index);
                idToWord.put(index, token);
                index++;
            }
        } catch (IOException e) {
            System.err.println("Failed to load vocabulary: " + e.getMessage());
        }
    }

    public int getId(String token) {
        return wordToId.getOrDefault(token, unkTokenId);
    }

    public String getToken(int id) {
        return idToWord.getOrDefault(id, "[UNK]");
    }

    public int size() {
        return wordToId.size();
    }
}