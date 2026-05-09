package SequenceProcessing.Bert.Tokenizer;

import java.util.ArrayList;

public class WordPieceTokenizer {
    private final Vocabulary vocab;
    private final int maxInputCharsPerWord;

    public WordPieceTokenizer(Vocabulary vocab) {
        this.vocab = vocab;
        this.maxInputCharsPerWord = 200;
    }

    public ArrayList<String> tokenize(String text) {
        ArrayList<String> outputTokens = new ArrayList<>();
        String[] words = text.split("\\s+");

        for (String word : words) {
            if (word.length() > maxInputCharsPerWord) {
                outputTokens.add("[UNK]");
                continue;
            }

            int start = 0;
            boolean isBad = false;
            ArrayList<String> subTokens = new ArrayList<>();

            while (start < word.length()) {
                int end = word.length();
                String curSubstring = null;

                while (start < end) {
                    String subStr = word.substring(start, end);
                    if (start > 0) {
                        subStr = "##" + subStr;
                    }
                    if (vocab.getId(subStr) != vocab.unkTokenId) {
                        curSubstring = subStr;
                        break;
                    }
                    end--;
                }

                if (curSubstring == null) {
                    isBad = true;
                    break;
                }

                subTokens.add(curSubstring);
                start = end;
            }

            if (isBad) {
                outputTokens.add("[UNK]");
            } else {
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;
    }
}