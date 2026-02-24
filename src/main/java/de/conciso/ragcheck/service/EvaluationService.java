package de.conciso.ragcheck.service;

import de.conciso.ragcheck.api.LightRagClient;
import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.GraphRetrievalRunResult;
import de.conciso.ragcheck.model.LlmRunResult;
import de.conciso.ragcheck.model.TestCase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class EvaluationService {

    private static final Logger log = LoggerFactory.getLogger(EvaluationService.class);

    private final LightRagClient lightRagClient;
    private final int runsPerTestCase;

    public EvaluationService(
            LightRagClient lightRagClient,
            @Value("${ragchecker.runs.per-testcase:1}") int runsPerTestCase
    ) {
        this.lightRagClient = lightRagClient;
        this.runsPerTestCase = runsPerTestCase;
    }

    public List<AggregatedEvalResult> evaluate(List<TestCase> testCases) {
        return testCases.stream().map(tc -> {
            List<GraphRetrievalRunResult> graphRuns = new ArrayList<>();
            List<LlmRunResult> llmRuns = new ArrayList<>();

            for (int i = 1; i <= runsPerTestCase; i++) {
                log.info("Running [{}] ({}/{}) — Graph: {}", tc.id(), i, runsPerTestCase, tc.prompt());
                long graphStart = System.currentTimeMillis();
                List<String> orderedDocs = lightRagClient.queryData(tc.prompt());
                long graphDuration = System.currentTimeMillis() - graphStart;
                graphRuns.add(GraphRetrievalRunResult.of(tc, orderedDocs, graphDuration));
                log.info("Running [{}] ({}/{}) — Graph fertig in {}s", tc.id(), i, runsPerTestCase, String.format("%.2f", graphDuration / 1000.0));

                log.info("Running [{}] ({}/{}) — LLM:   {}", tc.id(), i, runsPerTestCase, tc.prompt());
                long llmStart = System.currentTimeMillis();
                LightRagClient.LlmResponse llmResponse = lightRagClient.queryLlm(tc.prompt());
                long llmDuration = System.currentTimeMillis() - llmStart;
                llmRuns.add(LlmRunResult.of(tc, llmResponse.referencedDocuments(), llmResponse.responseText(), llmDuration));
                log.info("Running [{}] ({}/{}) — LLM fertig in {}s", tc.id(), i, runsPerTestCase, String.format("%.2f", llmDuration / 1000.0));
            }

            AggregatedEvalResult result = AggregatedEvalResult.of(tc, graphRuns, llmRuns);

            log.info("[{}] Graph — MRR={} NDCG={} Recall@k={} ØDauer={}s",
                    result.testCaseId(),
                    fmt(result.graphMetrics().avgMrr()),
                    fmt(result.graphMetrics().avgNdcgAtK()),
                    fmt(result.graphMetrics().avgRecallAtK()),
                    String.format("%.2f", result.graphMetrics().avgDurationMs() / 1000.0));
            log.info("[{}] LLM   — Recall={} Precision={} F1={} Hit={} MRR={} ØDauer={}s",
                    result.testCaseId(),
                    fmt(result.llmMetrics().avgRecall()),
                    fmt(result.llmMetrics().avgPrecision()),
                    fmt(result.llmMetrics().avgF1()),
                    fmt(result.llmMetrics().hitRate()),
                    fmt(result.llmMetrics().avgMrr()),
                    String.format("%.2f", result.llmMetrics().avgDurationMs() / 1000.0));

            return result;
        }).toList();
    }

    private String fmt(double v) {
        return String.format("%.2f", v);
    }
}
