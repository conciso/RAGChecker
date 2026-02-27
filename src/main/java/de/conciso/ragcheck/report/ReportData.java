package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;

import java.time.Instant;
import java.util.Arrays;
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
        double stdDevGraphMrr,
        double avgGraphNdcgAtK,
        double stdDevGraphNdcgAtK,
        double avgGraphRecallAtK,
        double stdDevGraphRecallAtK,
        // LLM summary
        double avgLlmRecall,
        double stdDevLlmRecall,
        double avgLlmPrecision,
        double stdDevLlmPrecision,
        double avgLlmF1,
        double stdDevLlmF1,
        double avgLlmHitRate,
        double stdDevLlmHitRate,
        double avgLlmMrr,
        double stdDevLlmMrr
) {
    public static ReportData of(List<AggregatedEvalResult> results,
                                String runLabel, Map<String, String> runParameters,
                                List<String> queryModes, int topK, int runsPerTestCase,
                                String testCasesPath) {
        double[] graphMrr      = results.stream().mapToDouble(r -> r.graphMetrics().avgMrr()).toArray();
        double[] graphNdcg     = results.stream().mapToDouble(r -> r.graphMetrics().avgNdcgAtK()).toArray();
        double[] graphRecall   = results.stream().mapToDouble(r -> r.graphMetrics().avgRecallAtK()).toArray();
        double[] llmRecall     = results.stream().mapToDouble(r -> r.llmMetrics().avgRecall()).toArray();
        double[] llmPrecision  = results.stream().mapToDouble(r -> r.llmMetrics().avgPrecision()).toArray();
        double[] llmF1         = results.stream().mapToDouble(r -> r.llmMetrics().avgF1()).toArray();
        double[] llmHitRate    = results.stream().mapToDouble(r -> r.llmMetrics().hitRate()).toArray();
        double[] llmMrr        = results.stream().mapToDouble(r -> r.llmMetrics().avgMrr()).toArray();

        return new ReportData(Instant.now(), runLabel, runParameters, queryModes, topK,
                runsPerTestCase, testCasesPath, results,
                mean(graphMrr),     stdDev(graphMrr),
                mean(graphNdcg),    stdDev(graphNdcg),
                mean(graphRecall),  stdDev(graphRecall),
                mean(llmRecall),    stdDev(llmRecall),
                mean(llmPrecision), stdDev(llmPrecision),
                mean(llmF1),        stdDev(llmF1),
                mean(llmHitRate),   stdDev(llmHitRate),
                mean(llmMrr),       stdDev(llmMrr));
    }

    private static double mean(double[] values) {
        return Arrays.stream(values).average().orElse(0.0);
    }

    static double stdDev(double[] values) {
        if (values.length < 2) return 0.0;
        double mean = mean(values);
        double variance = Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(0.0);
        return Math.sqrt(variance);
    }
}
