package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.EvalResult;

import java.time.Instant;
import java.util.List;

public record ReportData(
        Instant timestamp,
        String queryMode,
        int topK,
        String testCasesPath,
        List<EvalResult> results,
        double avgRecall,
        double avgPrecision,
        double avgF1
) {
    public static ReportData of(List<EvalResult> results, String queryMode, int topK, String testCasesPath) {
        double avgRecall    = results.stream().mapToDouble(EvalResult::recall).average().orElse(0.0);
        double avgPrecision = results.stream().mapToDouble(EvalResult::precision).average().orElse(0.0);
        double avgF1        = results.stream().mapToDouble(EvalResult::f1).average().orElse(0.0);
        return new ReportData(Instant.now(), queryMode, topK, testCasesPath, results, avgRecall, avgPrecision, avgF1);
    }
}
