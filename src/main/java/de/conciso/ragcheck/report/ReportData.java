package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;

import java.time.Instant;
import java.util.List;
import java.util.Map;

public record ReportData(
        Instant timestamp,
        String runLabel,
        Map<String, String> runParameters,
        List<String> queryModes,
        int topK,
        int runsPerTestCase,
        String testCasesPath,
        List<AggregatedEvalResult> results,
        // Graph summary
        double avgGraphMrr,
        double avgGraphNdcgAtK,
        double avgGraphRecallAtK,
        // LLM summary
        double avgLlmRecall,
        double avgLlmPrecision,
        double avgLlmF1,
        double avgLlmHitRate,
        double avgLlmMrr
) {
    public static ReportData of(List<AggregatedEvalResult> results,
                                String runLabel, Map<String, String> runParameters,
                                List<String> queryModes, int topK, int runsPerTestCase,
                                String testCasesPath) {
        double avgGraphMrr      = results.stream().mapToDouble(r -> r.graphMetrics().avgMrr()).average().orElse(0.0);
        double avgGraphNdcg     = results.stream().mapToDouble(r -> r.graphMetrics().avgNdcgAtK()).average().orElse(0.0);
        double avgGraphRecall   = results.stream().mapToDouble(r -> r.graphMetrics().avgRecallAtK()).average().orElse(0.0);
        double avgLlmRecall     = results.stream().mapToDouble(r -> r.llmMetrics().avgRecall()).average().orElse(0.0);
        double avgLlmPrecision  = results.stream().mapToDouble(r -> r.llmMetrics().avgPrecision()).average().orElse(0.0);
        double avgLlmF1         = results.stream().mapToDouble(r -> r.llmMetrics().avgF1()).average().orElse(0.0);
        double avgLlmHitRate    = results.stream().mapToDouble(r -> r.llmMetrics().hitRate()).average().orElse(0.0);
        double avgLlmMrr        = results.stream().mapToDouble(r -> r.llmMetrics().avgMrr()).average().orElse(0.0);

        return new ReportData(Instant.now(), runLabel, runParameters, queryModes, topK,
                runsPerTestCase, testCasesPath, results,
                avgGraphMrr, avgGraphNdcg, avgGraphRecall,
                avgLlmRecall, avgLlmPrecision, avgLlmF1, avgLlmHitRate, avgLlmMrr);
    }
}
