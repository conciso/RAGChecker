package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;

import java.time.Instant;
import java.util.List;

public record ReportData(
        Instant timestamp,
        String queryMode,
        int topK,
        int runsPerTestCase,
        String testCasesPath,
        List<AggregatedEvalResult> results,
        double avgRecall,
        double avgPrecision,
        double avgF1,
        double avgHitRate,
        double avgMrr
) {
    public static ReportData of(List<AggregatedEvalResult> results, String queryMode, int topK,
                                int runsPerTestCase, String testCasesPath) {
        double avgRecall    = results.stream().mapToDouble(AggregatedEvalResult::avgRecall).average().orElse(0.0);
        double avgPrecision = results.stream().mapToDouble(AggregatedEvalResult::avgPrecision).average().orElse(0.0);
        double avgF1        = results.stream().mapToDouble(AggregatedEvalResult::avgF1).average().orElse(0.0);
        double avgHitRate   = results.stream().mapToDouble(AggregatedEvalResult::hitRate).average().orElse(0.0);
        double avgMrr       = results.stream().mapToDouble(AggregatedEvalResult::mrr).average().orElse(0.0);
        return new ReportData(Instant.now(), queryMode, topK, runsPerTestCase, testCasesPath,
                results, avgRecall, avgPrecision, avgF1, avgHitRate, avgMrr);
    }
}
