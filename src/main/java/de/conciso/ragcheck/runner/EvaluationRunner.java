package de.conciso.ragcheck.runner;

import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.TestCase;
import de.conciso.ragcheck.report.ReportWriter;
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
    private final ReportWriter reportWriter;

    public EvaluationRunner(TestCaseLoader loader, EvaluationService evaluationService, ReportWriter reportWriter) {
        this.loader = loader;
        this.evaluationService = evaluationService;
        this.reportWriter = reportWriter;
    }

    @Override
    public void run(String... args) throws Exception {
        List<TestCase> testCases = loader.load();
        log.info("Loaded {} test case(s)", testCases.size());

        List<AggregatedEvalResult> results = evaluationService.evaluate(testCases);

        printSummary(results);
        reportWriter.write(results);
    }

    private void printSummary(List<AggregatedEvalResult> results) {
        String separator = "-".repeat(82);
        System.out.println();
        System.out.println("=== RAGChecker Summary ===");
        System.out.printf("%-12s %8s %8s %8s %8s %8s %8s %8s %8s%n",
                "ID", "Recall", "±σ", "Prec.", "±σ", "F1", "±σ", "Hit%", "MRR");
        System.out.println(separator);

        for (AggregatedEvalResult r : results) {
            System.out.printf("%-12s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f%n",
                    r.testCaseId(),
                    r.avgRecall(),    r.stdDevRecall(),
                    r.avgPrecision(), r.stdDevPrecision(),
                    r.avgF1(),        r.stdDevF1(),
                    r.hitRate(),      r.mrr());
        }

        double avgRecall    = results.stream().mapToDouble(AggregatedEvalResult::avgRecall).average().orElse(0.0);
        double avgPrecision = results.stream().mapToDouble(AggregatedEvalResult::avgPrecision).average().orElse(0.0);
        double avgF1        = results.stream().mapToDouble(AggregatedEvalResult::avgF1).average().orElse(0.0);
        double avgHitRate   = results.stream().mapToDouble(AggregatedEvalResult::hitRate).average().orElse(0.0);
        double avgMrr       = results.stream().mapToDouble(AggregatedEvalResult::mrr).average().orElse(0.0);

        System.out.println(separator);
        System.out.printf("%-12s %8.2f %8s %8.2f %8s %8.2f %8s %8.2f %8.2f%n",
                "AVG", avgRecall, "", avgPrecision, "", avgF1, "", avgHitRate, avgMrr);
        System.out.println();
    }
}
