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
                List<String> orderedDocs = lightRagClient.queryData(tc.prompt());
                graphRuns.add(GraphRetrievalRunResult.of(tc, orderedDocs));

                log.info("Running [{}] ({}/{}) — LLM:   {}", tc.id(), i, runsPerTestCase, tc.prompt());
                LightRagClient.LlmResponse llmResponse = lightRagClient.queryLlm(tc.prompt());
                llmRuns.add(LlmRunResult.of(tc, llmResponse.referencedDocuments(), llmResponse.responseText()));
            }

            AggregatedEvalResult result = AggregatedEvalResult.of(tc, graphRuns, llmRuns);

            log.info("[{}] Graph — MRR={} NDCG={} Recall@k={}",
                    result.testCaseId(),
                    fmt(result.graphMetrics().avgMrr()),
                    fmt(result.graphMetrics().avgNdcgAtK()),
                    fmt(result.graphMetrics().avgRecallAtK()));
            log.info("[{}] LLM   — Recall={} Precision={} F1={} Hit={} MRR={}",
                    result.testCaseId(),
                    fmt(result.llmMetrics().avgRecall()),
                    fmt(result.llmMetrics().avgPrecision()),
                    fmt(result.llmMetrics().avgF1()),
                    fmt(result.llmMetrics().hitRate()),
                    fmt(result.llmMetrics().avgMrr()));

            return result;
        }).toList();
    }

    private String fmt(double v) {
        return String.format("%.2f", v);
    }
}
