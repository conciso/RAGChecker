package de.conciso.ragcheck.model;

import java.util.List;

public record AggregatedLlmMetrics(
        double avgRecall,    double stdDevRecall,
        double avgPrecision, double stdDevPrecision,
        double avgF1,        double stdDevF1,
        double hitRate,
        double avgMrr,       double stdDevMrr
) {
    public static AggregatedLlmMetrics of(List<LlmRunResult> runs) {
        double[] recalls    = runs.stream().mapToDouble(LlmRunResult::recall).toArray();
        double[] precisions = runs.stream().mapToDouble(LlmRunResult::precision).toArray();
        double[] f1s        = runs.stream().mapToDouble(LlmRunResult::f1).toArray();
        double[] hits       = runs.stream().mapToDouble(r -> r.hit() ? 1.0 : 0.0).toArray();
        double[] mrrs       = runs.stream().mapToDouble(LlmRunResult::mrr).toArray();

        return new AggregatedLlmMetrics(
                mean(recalls),    stdDev(recalls),
                mean(precisions), stdDev(precisions),
                mean(f1s),        stdDev(f1s),
                mean(hits),
                mean(mrrs),       stdDev(mrrs)
        );
    }

    private static double mean(double[] v) {
        if (v.length == 0) return 0.0;
        double s = 0;
        for (double x : v) s += x;
        return s / v.length;
    }

    private static double stdDev(double[] v) {
        if (v.length < 2) return 0.0;
        double avg = mean(v);
        double sq = 0;
        for (double x : v) sq += (x - avg) * (x - avg);
        return Math.sqrt(sq / v.length);
    }
}
