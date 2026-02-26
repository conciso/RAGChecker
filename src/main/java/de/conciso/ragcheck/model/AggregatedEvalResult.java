package de.conciso.ragcheck.model;

import java.util.List;

public record AggregatedEvalResult(
        String testCaseId,
        String prompt,
        List<String> expectedDocuments,
        int runs,
        String queryMode,
        AggregatedGraphMetrics graphMetrics,
        AggregatedLlmMetrics llmMetrics,
        List<GraphRetrievalRunResult> graphRuns,
        List<LlmRunResult> llmRuns
) {
    public static AggregatedEvalResult of(TestCase tc,
                                          List<GraphRetrievalRunResult> graphRuns,
                                          List<LlmRunResult> llmRuns,
                                          String queryMode) {
        return new AggregatedEvalResult(
                tc.id(), tc.prompt(), tc.expectedDocuments(),
                graphRuns.size(), queryMode,
                AggregatedGraphMetrics.of(tc.expectedDocuments(), graphRuns),
                AggregatedLlmMetrics.of(llmRuns),
                graphRuns, llmRuns
        );
    }
}
