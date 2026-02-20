package de.conciso.ragcheck.service;

import de.conciso.ragcheck.api.LightRagClient;
import de.conciso.ragcheck.model.EvalResult;
import de.conciso.ragcheck.model.QueryResult;
import de.conciso.ragcheck.model.TestCase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EvaluationService {

    private static final Logger log = LoggerFactory.getLogger(EvaluationService.class);

    private final LightRagClient lightRagClient;

    public EvaluationService(LightRagClient lightRagClient) {
        this.lightRagClient = lightRagClient;
    }

    public List<EvalResult> evaluate(List<TestCase> testCases) {
        return testCases.stream().map(tc -> {
            log.info("Running [{}]: {}", tc.id(), tc.prompt());
            QueryResult queryResult = lightRagClient.query(tc.prompt());
            EvalResult result = EvalResult.of(tc, queryResult);
            log.info("[{}] recall={} precision={} f1={}",
                    result.testCaseId(),
                    String.format("%.2f", result.recall()),
                    String.format("%.2f", result.precision()),
                    String.format("%.2f", result.f1()));
            return result;
        }).toList();
    }
}
