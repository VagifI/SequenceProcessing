package SequenceProcessing.Bert.Tokenizer;

import java.util.ArrayList;
// import MorphologicalAnalysis.MorphologicalParse;
// import MorphologicalAnalysis.FsmMorphologicalAnalyzer;

public class HybridTokenizer {
    private final Vocabulary vocab;
    private final WordPieceTokenizer wordPieceFallback;

    public HybridTokenizer(Vocabulary vocab) {
        this.vocab = vocab;
        this.wordPieceFallback = new WordPieceTokenizer(vocab);
    }

    public ArrayList<Integer> encode(String text) {
        ArrayList<Integer> tokenIds = new ArrayList<>();
        tokenIds.add(vocab.getId("[CLS]")); // Начало предложения

        String[] rawWords = text.split("\\s+");

        for (String word : rawWords) {
            ArrayList<String> morphemes = performMorphologicalAnalysis(word);

            if (morphemes != null && !morphemes.isEmpty()) {
                for (String morpheme : morphemes) {
                    tokenIds.add(vocab.getId(morpheme));
                }
            } else {
                ArrayList<String> wordPieceTokens = wordPieceFallback.tokenize(word);
                for (String wpToken : wordPieceTokens) {
                    tokenIds.add(vocab.getId(wpToken));
                }
            }
        }

        tokenIds.add(vocab.getId("[SEP]"));
        return tokenIds;
    }

    private ArrayList<String> performMorphologicalAnalysis(String word) {

        return null;
    }
}