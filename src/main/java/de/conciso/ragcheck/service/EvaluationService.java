package de.conciso.ragcheck.service;

import de.conciso.ragcheck.api.LightRagClient;
import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.GraphRetrievalData;
import de.conciso.ragcheck.model.GraphRetrievalRunResult;
import de.conciso.ragcheck.model.LlmRunResult;
import de.conciso.ragcheck.model.TestCase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class EvaluationService {

    private static final Logger log = LoggerFactory.getLogger(EvaluationService.class);

    private final LightRagClient lightRagClient;
    private final int runsPerTestCase;
    private final List<String> queryModes;

    public EvaluationService(
            LightRagClient lightRagClient,
            @Value("${ragchecker.runs.per-testcase:1}") int runsPerTestCase,
            @Value("${ragchecker.query.modes}") String queryModesStr
    ) {
        this.lightRagClient = lightRagClient;
        this.runsPerTestCase = runsPerTestCase;
        this.queryModes = List.of(queryModesStr.split(","));
    }

    public List<AggregatedEvalResult> evaluate(List<TestCase> testCases) {
        List<AggregatedEvalResult> results = new ArrayList<>();
        List<String> failures = new ArrayList<>();

        for (TestCase tc : testCases) {
            for (String mode : queryModes) {
                List<GraphRetrievalRunResult> graphRuns = new ArrayList<>();
                List<LlmRunResult> llmRuns = new ArrayList<>();

                for (int i = 1; i <= runsPerTestCase; i++) {
                    log.info("Running [{}/{}] ({}/{}) — Graph: {}", tc.id(), mode, i, runsPerTestCase, tc.prompt());
                    long graphStart = System.currentTimeMillis();
                    GraphRetrievalData graphData;
                    try {
                        graphData = lightRagClient.queryData(tc.prompt(), mode);
                    } catch (RestClientException e) {
                        String failure = String.format("Graph [%s/%s] Lauf %d/%d: %s", tc.id(), mode, i, runsPerTestCase, e.getMessage());
                        log.error("API-Fehler — {}", failure);
                        failures.add(failure);
                        graphData = new GraphRetrievalData(List.of(), List.of(), List.of(), Map.of());
                    }
                    long graphDuration = System.currentTimeMillis() - graphStart;
                    graphRuns.add(GraphRetrievalRunResult.of(tc, graphData, graphDuration));
                    log.info("Running [{}/{}] ({}/{}) — Graph fertig in {}s", tc.id(), mode, i, runsPerTestCase, String.format("%.2f", graphDuration / 1000.0));

                    log.info("Running [{}/{}] ({}/{}) — LLM:   {}", tc.id(), mode, i, runsPerTestCase, tc.prompt());
                    long llmStart = System.currentTimeMillis();
                    LightRagClient.LlmResponse llmResponse;
                    try {
                        llmResponse = lightRagClient.queryLlm(tc.prompt(), mode);
                    } catch (RestClientException e) {
                        String failure = String.format("LLM   [%s/%s] Lauf %d/%d: %s", tc.id(), mode, i, runsPerTestCase, e.getMessage());
                        log.error("API-Fehler — {}", failure);
                        failures.add(failure);
                        llmResponse = new LightRagClient.LlmResponse(null, List.of());
                    }
                    long llmDuration = System.currentTimeMillis() - llmStart;
                    llmRuns.add(LlmRunResult.of(tc, llmResponse.referencedDocuments(), llmResponse.responseText(), llmDuration));
                    log.info("Running [{}/{}] ({}/{}) — LLM fertig in {}s", tc.id(), mode, i, runsPerTestCase, String.format("%.2f", llmDuration / 1000.0));
                }

                AggregatedEvalResult result = AggregatedEvalResult.of(tc, graphRuns, llmRuns, mode);

                log.info("[{}/{}] Graph — MRR={} NDCG={} Recall@k={} ØDauer={}s",
                        result.testCaseId(), result.queryMode(),
                        fmt(result.graphMetrics().avgMrr()),
                        fmt(result.graphMetrics().avgNdcgAtK()),
                        fmt(result.graphMetrics().avgRecallAtK()),
                        String.format("%.2f", result.graphMetrics().avgDurationMs() / 1000.0));
                log.info("[{}/{}] LLM   — Recall={} Precision={} F1={} Hit={} MRR={} ØDauer={}s",
                        result.testCaseId(), result.queryMode(),
                        fmt(result.llmMetrics().avgRecall()),
                        fmt(result.llmMetrics().avgPrecision()),
                        fmt(result.llmMetrics().avgF1()),
                        fmt(result.llmMetrics().hitRate()),
                        fmt(result.llmMetrics().avgMrr()),
                        String.format("%.2f", result.llmMetrics().avgDurationMs() / 1000.0));

                results.add(result);
            }
        }
        if (!failures.isEmpty()) {
            throw new PartialEvaluationException(results, failures);
        }
        return results;
    }

    public static class PartialEvaluationException extends RuntimeException {
        private final List<AggregatedEvalResult> partialResults;
        private final List<String> failures;

        public PartialEvaluationException(List<AggregatedEvalResult> partialResults, List<String> failures) {
            super(failures.size() + " API-Aufruf(e) fehlgeschlagen:\n  " + String.join("\n  ", failures));
            this.partialResults = partialResults;
            this.failures = failures;
        }

        public List<AggregatedEvalResult> getPartialResults() { return partialResults; }
        public List<String> getFailures() { return failures; }
    }

    private String fmt(double v) {
        return String.format("%.2f", v);
    }
}
