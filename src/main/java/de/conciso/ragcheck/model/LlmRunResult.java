package de.conciso.ragcheck.model;

import java.util.List;

public record LlmRunResult(
        List<String> retrievedDocuments,
        String responseText,
        double recall,
        double precision,
        double f1,
        boolean hit,
        double mrr
) {
    public static LlmRunResult of(TestCase tc, List<String> referencedDocuments, String responseText) {
        List<String> expected  = tc.expectedDocuments();
        List<String> retrieved = referencedDocuments;

        long hits = expected.stream()
                .filter(exp -> retrieved.stream()
                        .anyMatch(ret -> ret.equalsIgnoreCase(exp)
                                || ret.toLowerCase().contains(exp.toLowerCase())))
                .count();

        double recall    = expected.isEmpty()  ? 0.0 : (double) hits / expected.size();
        double precision = retrieved.isEmpty() ? 0.0 : (double) hits / retrieved.size();
        double f1        = (recall + precision) == 0.0 ? 0.0
                : 2.0 * recall * precision / (recall + precision);
        boolean hit      = hits > 0;
        double mrr       = computeMrr(expected, retrieved);

        return new LlmRunResult(retrieved, responseText, recall, precision, f1, hit, mrr);
    }

    private static double computeMrr(List<String> expected, List<String> retrieved) {
        for (int i = 0; i < retrieved.size(); i++) {
            String ret = retrieved.get(i);
            for (String exp : expected) {
                if (ret.equalsIgnoreCase(exp) || ret.toLowerCase().contains(exp.toLowerCase())) {
                    return 1.0 / (i + 1);
                }
            }
        }
        return 0.0;
    }
}
