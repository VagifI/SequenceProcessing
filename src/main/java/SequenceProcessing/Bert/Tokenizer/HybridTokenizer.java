package SequenceProcessing.Bert.Tokenizer;

import java.util.ArrayList;
import MorphologicalAnalysis.FsmMorphologicalAnalyzer;
import MorphologicalAnalysis.FsmParseList;
import MorphologicalAnalysis.FsmParse;
import MorphologicalAnalysis.MorphologicalParse;
import SequenceProcessing.Bert.Tokenizer.Vocabulary;
import SequenceProcessing.Bert.Tokenizer.WordPieceTokenizer;

public class HybridTokenizer {
    private final Vocabulary vocab;
    private final WordPieceTokenizer wordPieceFallback;
    private FsmMorphologicalAnalyzer analyzer;

    public HybridTokenizer(Vocabulary vocab) {
        this.vocab = vocab;
        this.wordPieceFallback = new WordPieceTokenizer(vocab);

        try {
            this.analyzer = new FsmMorphologicalAnalyzer();
        } catch (Exception e) {
            System.out.println("Предупреждение: Не удалось загрузить MorphologicalAnalyzer.");
        }
    }

    public ArrayList<Integer> encode(String text) {
        ArrayList<Integer> tokenIds = new ArrayList<>();
        tokenIds.add(vocab.getId("[CLS]"));

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
        if (this.analyzer == null) {
            return null;
        }

        try {
            FsmParseList parses = analyzer.morphologicalAnalysis(word);

            if (parses.size() > 0) {
                FsmParse bestParse = parses.getFsmParse(0);
                ArrayList<String> tokens = new ArrayList<>();

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
        } catch (Exception e) {
            return null;
        }

        return null;
    }
}