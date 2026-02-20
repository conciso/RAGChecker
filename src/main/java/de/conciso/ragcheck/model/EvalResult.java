package de.conciso.ragcheck.model;

import java.util.List;

public record EvalResult(
        String testCaseId,
        String prompt,
        List<String> expectedDocuments,
        List<String> retrievedDocuments,
        double recall,
        double precision,
        double f1,
        QueryResult queryResult
) {
    public static EvalResult of(TestCase tc, QueryResult queryResult) {
        List<String> expected = tc.expectedDocuments();
        List<String> retrieved = queryResult.referencedFileNames();

        long hits = expected.stream()
                .filter(exp -> retrieved.stream()
                        .anyMatch(ret -> ret.equalsIgnoreCase(exp) || ret.toLowerCase().contains(exp.toLowerCase())))
                .count();

        double recall    = expected.isEmpty() ? 0.0 : (double) hits / expected.size();
        double precision = retrieved.isEmpty() ? 0.0 : (double) hits / retrieved.size();
        double f1        = (recall + precision) == 0.0 ? 0.0
                : 2.0 * recall * precision / (recall + precision);

        return new EvalResult(tc.id(), tc.prompt(), expected, retrieved, recall, precision, f1, queryResult);
    }
}
