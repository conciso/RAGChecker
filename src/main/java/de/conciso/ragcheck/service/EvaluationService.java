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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

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

    /**
     * Wertet alle Testfälle aus.
     * Pro Mode und Lauf werden alle Testfälle in einer gemeinsamen Session gestellt
     * (conversation_history wächst mit jeder Frage innerhalb der Session).
     * Nach jeder abgeschlossenen Session wird onSessionComplete mit den bisher
     * akkumulierten Zwischenergebnissen aufgerufen.
     * Wirft PartialEvaluationException wenn mindestens ein API-Aufruf fehlgeschlagen ist.
     */
    public List<AggregatedEvalResult> evaluate(List<TestCase> testCases,
                                               List<String> failures,
                                               Consumer<List<AggregatedEvalResult>> onSessionComplete) {
        // Akkumulatoren: Key = "tcId::mode"
        Map<String, List<GraphRetrievalRunResult>> graphRunsMap = new LinkedHashMap<>();
        Map<String, List<LlmRunResult>> llmRunsMap = new LinkedHashMap<>();
        for (TestCase tc : testCases) {
            for (String mode : queryModes) {
                String key = key(tc.id(), mode);
                graphRunsMap.put(key, new ArrayList<>());
                llmRunsMap.put(key, new ArrayList<>());
            }
        }

        for (String mode : queryModes) {
            for (int run = 1; run <= runsPerTestCase; run++) {
                log.info("=== Session: mode={} Lauf={}/{} ===", mode, run, runsPerTestCase);
                List<LightRagClient.ConversationEntry> history = new ArrayList<>();

                for (TestCase tc : testCases) {
                    String key = key(tc.id(), mode);

                    // --- Graph ---
                    log.info("  [{}/{}] ({}/{}) — Graph: {}", tc.id(), mode, run, runsPerTestCase, tc.prompt());
                    long graphStart = System.currentTimeMillis();
                    GraphRetrievalData graphData;
                    try {
                        graphData = lightRagClient.queryData(tc.prompt(), mode, history);
                    } catch (RestClientException e) {
                        String failure = String.format("Graph [%s/%s] Lauf %d/%d: %s", tc.id(), mode, run, runsPerTestCase, e.getMessage());
                        log.error("API-Fehler — {}", failure);
                        failures.add(failure);
                        graphData = new GraphRetrievalData(List.of(), List.of(), List.of(), Map.of());
                    }
                    long graphDuration = System.currentTimeMillis() - graphStart;
                    graphRunsMap.get(key).add(GraphRetrievalRunResult.of(tc, graphData, graphDuration));
                    log.info("  [{}/{}] ({}/{}) — Graph fertig in {}s", tc.id(), mode, run, runsPerTestCase,
                            String.format("%.2f", graphDuration / 1000.0));

                    // --- LLM ---
                    log.info("  [{}/{}] ({}/{}) — LLM:   {}", tc.id(), mode, run, runsPerTestCase, tc.prompt());
                    long llmStart = System.currentTimeMillis();
                    LightRagClient.LlmResponse llmResponse;
                    try {
                        llmResponse = lightRagClient.queryLlm(tc.prompt(), mode, history);
                    } catch (RestClientException e) {
                        String failure = String.format("LLM   [%s/%s] Lauf %d/%d: %s", tc.id(), mode, run, runsPerTestCase, e.getMessage());
                        log.error("API-Fehler — {}", failure);
                        failures.add(failure);
                        llmResponse = new LightRagClient.LlmResponse(null, List.of());
                    }
                    long llmDuration = System.currentTimeMillis() - llmStart;
                    llmRunsMap.get(key).add(LlmRunResult.of(tc, llmResponse.referencedDocuments(), llmResponse.responseText(), llmDuration));
                    log.info("  [{}/{}] ({}/{}) — LLM fertig in {}s", tc.id(), mode, run, runsPerTestCase,
                            String.format("%.2f", llmDuration / 1000.0));

                    // History für nächste Frage in dieser Session erweitern
                    history.add(new LightRagClient.ConversationEntry("user", tc.prompt()));
                    String assistantContent = llmResponse.responseText() != null ? llmResponse.responseText() : "";
                    history.add(new LightRagClient.ConversationEntry("assistant", assistantContent));
                }

                // Zwischenergebnis nach jeder Session
                List<AggregatedEvalResult> partial = buildResults(testCases, graphRunsMap, llmRunsMap);
                logSessionSummary(partial, mode, run);
                onSessionComplete.accept(partial);
            }
        }

        List<AggregatedEvalResult> results = buildResults(testCases, graphRunsMap, llmRunsMap);
        if (!failures.isEmpty()) {
            throw new PartialEvaluationException(results, failures);
        }
        return results;
    }

    // -------------------------------------------------------------------------

    private List<AggregatedEvalResult> buildResults(List<TestCase> testCases,
                                                    Map<String, List<GraphRetrievalRunResult>> graphRunsMap,
                                                    Map<String, List<LlmRunResult>> llmRunsMap) {
        List<AggregatedEvalResult> results = new ArrayList<>();
        for (TestCase tc : testCases) {
            for (String mode : queryModes) {
                String key = key(tc.id(), mode);
                List<GraphRetrievalRunResult> graphRuns = graphRunsMap.get(key);
                List<LlmRunResult> llmRuns = llmRunsMap.get(key);
                if (graphRuns == null || graphRuns.isEmpty()) continue;
                AggregatedEvalResult result = AggregatedEvalResult.of(tc, graphRuns, llmRuns, mode);
                results.add(result);
            }
        }
        return results;
    }

    private void logSessionSummary(List<AggregatedEvalResult> results, String mode, int run) {
        results.stream()
                .filter(r -> r.queryMode().equals(mode))
                .filter(r -> r.runs() == run)
                .forEach(r -> {
                    log.info("[{}/{}] Graph — MRR={} NDCG={} Recall@k={}",
                            r.testCaseId(), r.queryMode(),
                            fmt(r.graphMetrics().avgMrr()),
                            fmt(r.graphMetrics().avgNdcgAtK()),
                            fmt(r.graphMetrics().avgRecallAtK()));
                    log.info("[{}/{}] LLM   — Recall={} Precision={} F1={}",
                            r.testCaseId(), r.queryMode(),
                            fmt(r.llmMetrics().avgRecall()),
                            fmt(r.llmMetrics().avgPrecision()),
                            fmt(r.llmMetrics().avgF1()));
                });
    }

    private static String key(String tcId, String mode) {
        return tcId + "::" + mode;
    }

    private String fmt(double v) {
        return String.format("%.2f", v);
    }

    // -------------------------------------------------------------------------

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
}
