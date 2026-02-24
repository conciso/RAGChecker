package de.conciso.ragcheck.model;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public record GraphRetrievalRunResult(
        List<String> retrievedDocuments,
        Map<String, Integer> expectedDocumentRanks,
        double mrr,
        double ndcgAtK,
        double recallAtK,
        long durationMs
) {
    public static GraphRetrievalRunResult of(TestCase tc, List<String> orderedDocs, long durationMs) {
        List<String> expected = tc.expectedDocuments();

        Map<String, Integer> ranks = new LinkedHashMap<>();
        for (String exp : expected) {
            for (int i = 0; i < orderedDocs.size(); i++) {
                if (matches(orderedDocs.get(i), exp)) {
                    ranks.put(exp, i + 1);
                    break;
                }
            }
        }

        double mrr       = computeMrr(expected, orderedDocs);
        double ndcg      = computeNdcgAtK(expected, orderedDocs);
        double recallAtK = expected.isEmpty() ? 0.0 : (double) ranks.size() / expected.size();

        return new GraphRetrievalRunResult(orderedDocs, ranks, mrr, ndcg, recallAtK, durationMs);
    }

    public static boolean matches(String retrieved, String expected) {
        return retrieved.equalsIgnoreCase(expected)
                || retrieved.toLowerCase().contains(expected.toLowerCase());
    }

    private static double computeMrr(List<String> expected, List<String> retrieved) {
        for (int i = 0; i < retrieved.size(); i++) {
            for (String exp : expected) {
                if (matches(retrieved.get(i), exp)) return 1.0 / (i + 1);
            }
        }
        return 0.0;
    }

    private static double computeNdcgAtK(List<String> expected, List<String> retrieved) {
        double dcg = 0.0;
        for (int i = 0; i < retrieved.size(); i++) {
            String doc = retrieved.get(i);
            boolean relevant = expected.stream().anyMatch(exp -> matches(doc, exp));
            if (relevant) dcg += 1.0 / (Math.log(i + 2) / Math.log(2));
        }
        int numRelevant = Math.min(expected.size(), retrieved.size());
        double idcg = 0.0;
        for (int i = 0; i < numRelevant; i++) {
            idcg += 1.0 / (Math.log(i + 2) / Math.log(2));
        }
        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }
}
