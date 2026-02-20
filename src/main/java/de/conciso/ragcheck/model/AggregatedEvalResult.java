package de.conciso.ragcheck.model;

import java.util.List;

public record AggregatedEvalResult(
        String testCaseId,
        String prompt,
        List<String> expectedDocuments,
        int runs,
        double avgRecall,
        double avgPrecision,
        double avgF1,
        double stdDevRecall,
        double stdDevPrecision,
        double stdDevF1,
        double hitRate,
        double mrr,
        List<EvalResult> runResults
) {
    public static AggregatedEvalResult of(TestCase tc, List<EvalResult> runResults) {
        double[] recalls    = runResults.stream().mapToDouble(EvalResult::recall).toArray();
        double[] precisions = runResults.stream().mapToDouble(EvalResult::precision).toArray();
        double[] f1s        = runResults.stream().mapToDouble(EvalResult::f1).toArray();
        double[] hits       = runResults.stream().mapToDouble(r -> r.recall() > 0.0 ? 1.0 : 0.0).toArray();
        double[] rrs        = runResults.stream()
                .mapToDouble(r -> reciprocalRank(tc.expectedDocuments(), r.retrievedDocuments()))
                .toArray();

        return new AggregatedEvalResult(
                tc.id(), tc.prompt(), tc.expectedDocuments(),
                runResults.size(),
                mean(recalls), mean(precisions), mean(f1s),
                stdDev(recalls), stdDev(precisions), stdDev(f1s),
                mean(hits), mean(rrs),
                runResults
        );
    }

    private static double reciprocalRank(List<String> expected, List<String> retrieved) {
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

    private static double mean(double[] values) {
        if (values.length == 0) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.length;
    }

    private static double stdDev(double[] values) {
        if (values.length < 2) return 0.0;
        double avg = mean(values);
        double sumSq = 0.0;
        for (double v : values) sumSq += (v - avg) * (v - avg);
        return Math.sqrt(sumSq / values.length);
    }
}
