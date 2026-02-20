package de.conciso.ragcheck.runner;

import de.conciso.ragcheck.model.EvalResult;
import de.conciso.ragcheck.model.TestCase;
import de.conciso.ragcheck.service.EvaluationService;
import de.conciso.ragcheck.service.TestCaseLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class EvaluationRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(EvaluationRunner.class);

    private final TestCaseLoader loader;
    private final EvaluationService evaluationService;

    public EvaluationRunner(TestCaseLoader loader, EvaluationService evaluationService) {
        this.loader = loader;
        this.evaluationService = evaluationService;
    }

    @Override
    public void run(String... args) throws Exception {
        List<TestCase> testCases = loader.load();
        log.info("Loaded {} test case(s)", testCases.size());

        List<EvalResult> results = evaluationService.evaluate(testCases);

        printSummary(results);
    }

    private void printSummary(List<EvalResult> results) {
        String separator = "-".repeat(62);
        System.out.println();
        System.out.println("=== RAGChecker Summary ===");
        System.out.printf("%-12s %8s %10s %8s %10s %10s%n",
                "ID", "Recall", "Precision", "F1", "Retrieved", "Expected");
        System.out.println(separator);

        for (EvalResult r : results) {
            System.out.printf("%-12s %8.2f %10.2f %8.2f %10d %10d%n",
                    r.testCaseId(),
                    r.recall(),
                    r.precision(),
                    r.f1(),
                    r.retrievedDocuments().size(),
                    r.expectedDocuments().size());
        }

        double avgRecall    = results.stream().mapToDouble(EvalResult::recall).average().orElse(0.0);
        double avgPrecision = results.stream().mapToDouble(EvalResult::precision).average().orElse(0.0);
        double avgF1        = results.stream().mapToDouble(EvalResult::f1).average().orElse(0.0);
        int totalRetrieved  = results.stream().mapToInt(r -> r.retrievedDocuments().size()).sum();
        int totalExpected   = results.stream().mapToInt(r -> r.expectedDocuments().size()).sum();

        System.out.println(separator);
        System.out.printf("%-12s %8.2f %10.2f %8.2f %10d %10d%n",
                "AVG/TOTAL", avgRecall, avgPrecision, avgF1, totalRetrieved, totalExpected);
        System.out.println();
    }
}
