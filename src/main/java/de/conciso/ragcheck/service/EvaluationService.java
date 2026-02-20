package de.conciso.ragcheck.service;

import de.conciso.ragcheck.api.LightRagClient;
import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.EvalResult;
import de.conciso.ragcheck.model.QueryResult;
import de.conciso.ragcheck.model.TestCase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.IntStream;

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
            List<EvalResult> runs = IntStream.rangeClosed(1, runsPerTestCase)
                    .mapToObj(i -> {
                        log.info("Running [{}] ({}/{}) : {}", tc.id(), i, runsPerTestCase, tc.prompt());
                        QueryResult queryResult = lightRagClient.query(tc.prompt());
                        return EvalResult.of(tc, queryResult);
                    })
                    .toList();

            AggregatedEvalResult result = AggregatedEvalResult.of(tc, runs);
            log.info("[{}] avgRecall={} avgPrecision={} avgF1={} hitRate={} mrr={}",
                    result.testCaseId(),
                    String.format("%.2f", result.avgRecall()),
                    String.format("%.2f", result.avgPrecision()),
                    String.format("%.2f", result.avgF1()),
                    String.format("%.2f", result.hitRate()),
                    String.format("%.2f", result.mrr()));
            return result;
        }).toList();
    }
}
