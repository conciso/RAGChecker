package de.conciso.ragcheck.runner;

import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.GraphRetrievalRunResult;
import de.conciso.ragcheck.model.LlmRunResult;
import de.conciso.ragcheck.model.TestCase;
import de.conciso.ragcheck.report.ReportWriter;
import de.conciso.ragcheck.service.EvaluationService;
import de.conciso.ragcheck.service.TestCaseLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
public class EvaluationRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(EvaluationRunner.class);

    private final TestCaseLoader loader;
    private final EvaluationService evaluationService;
    private final ReportWriter reportWriter;

    public EvaluationRunner(TestCaseLoader loader, EvaluationService evaluationService,
                            ReportWriter reportWriter) {
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
        printDebug(results);
        reportWriter.write(results);
    }

    // -------------------------------------------------------------------------

    private void printSummary(List<AggregatedEvalResult> results) {
        String sep = "-".repeat(90);
        System.out.println();
        System.out.println("=== RAGChecker Summary ===");

        System.out.println();
        System.out.println("-- Graph-Retrieval (/query/data) --");
        System.out.printf("%-12s %8s %8s %10s %10s %12s %12s%n",
                "ID", "MRR", "±σ", "NDCG@k", "±σ", "Recall@k", "±σ");
        System.out.println(sep);
        for (AggregatedEvalResult r : results) {
            System.out.printf("%-12s %8.2f %8.2f %10.2f %10.2f %12.2f %12.2f%n",
                    r.testCaseId(),
                    r.graphMetrics().avgMrr(),       r.graphMetrics().stdDevMrr(),
                    r.graphMetrics().avgNdcgAtK(),   r.graphMetrics().stdDevNdcgAtK(),
                    r.graphMetrics().avgRecallAtK(), r.graphMetrics().stdDevRecallAtK());
        }
        System.out.println(sep);
        System.out.printf("%-12s %8.2f %8s %10.2f %10s %12.2f%n", "AVG",
                results.stream().mapToDouble(r -> r.graphMetrics().avgMrr()).average().orElse(0),
                "",
                results.stream().mapToDouble(r -> r.graphMetrics().avgNdcgAtK()).average().orElse(0),
                "",
                results.stream().mapToDouble(r -> r.graphMetrics().avgRecallAtK()).average().orElse(0));

        System.out.println();
        System.out.println("-- LLM (/query) --");
        System.out.printf("%-12s %8s %8s %10s %10s %8s %8s %8s %8s%n",
                "ID", "Recall", "±σ", "Precision", "±σ", "F1", "±σ", "Hit%", "MRR");
        System.out.println(sep);
        for (AggregatedEvalResult r : results) {
            System.out.printf("%-12s %8.2f %8.2f %10.2f %10.2f %8.2f %8.2f %8.2f %8.2f%n",
                    r.testCaseId(),
                    r.llmMetrics().avgRecall(),    r.llmMetrics().stdDevRecall(),
                    r.llmMetrics().avgPrecision(), r.llmMetrics().stdDevPrecision(),
                    r.llmMetrics().avgF1(),        r.llmMetrics().stdDevF1(),
                    r.llmMetrics().hitRate(),      r.llmMetrics().avgMrr());
        }
        System.out.println(sep);
        System.out.printf("%-12s %8.2f %8s %10.2f %8s %8.2f %8s %8.2f %8.2f%n", "AVG",
                results.stream().mapToDouble(r -> r.llmMetrics().avgRecall()).average().orElse(0), "",
                results.stream().mapToDouble(r -> r.llmMetrics().avgPrecision()).average().orElse(0), "",
                results.stream().mapToDouble(r -> r.llmMetrics().avgF1()).average().orElse(0), "",
                results.stream().mapToDouble(r -> r.llmMetrics().hitRate()).average().orElse(0),
                results.stream().mapToDouble(r -> r.llmMetrics().avgMrr()).average().orElse(0));
        System.out.println();
    }

    // -------------------------------------------------------------------------

    private void printDebug(List<AggregatedEvalResult> results) {
        String thick = "=".repeat(80);
        String thin  = "-".repeat(80);
        System.out.println(thick);
        System.out.println("=== RAGChecker Debug Output ===");
        System.out.println(thick);

        for (AggregatedEvalResult r : results) {
            System.out.println();
            System.out.println(thick);
            System.out.println("Testfall : " + r.testCaseId());
            System.out.println("Prompt   : " + r.prompt());
            System.out.println("Erwartet : " + String.join(", ", r.expectedDocuments()));
            System.out.println(thin);

            for (int i = 0; i < r.runs(); i++) {
                System.out.printf("=== Lauf %d/%d ===%n", i + 1, r.runs());

                // Graph
                GraphRetrievalRunResult g = r.graphRuns().get(i);
                System.out.println("  [Graph]");
                System.out.printf("    MRR=%.2f  NDCG@k=%.2f  Recall@k=%.2f%n",
                        g.mrr(), g.ndcgAtK(), g.recallAtK());
                System.out.println("    Ränge:");
                for (Map.Entry<String, Integer> e : g.expectedDocumentRanks().entrySet()) {
                    System.out.printf("      %-40s → Rang %d%n", e.getKey(), e.getValue());
                }
                for (String exp : r.expectedDocuments()) {
                    if (!g.expectedDocumentRanks().containsKey(exp)) {
                        System.out.printf("      %-40s → nicht gefunden%n", exp);
                    }
                }
                System.out.println("    Gefunden (alle): " + String.join(", ", g.retrievedDocuments()));

                // LLM
                LlmRunResult l = r.llmRuns().get(i);
                System.out.println("  [LLM]");
                System.out.printf("    Recall=%.2f  Precision=%.2f  F1=%.2f  Hit=%s  MRR=%.2f%n",
                        l.recall(), l.precision(), l.f1(), l.hit() ? "✓" : "✗", l.mrr());
                System.out.println("    Gefunden: " +
                        (l.retrievedDocuments().isEmpty() ? "(keine)" : String.join(", ", l.retrievedDocuments())));
                System.out.println("    LLM-Antwort:");
                if (l.responseText() == null || l.responseText().isBlank()) {
                    System.out.println("      (keine Antwort)");
                } else {
                    for (String line : l.responseText().split("\n")) {
                        System.out.println("      " + line);
                    }
                }
                System.out.println(thin);
            }

            System.out.printf("  Aggregiert Graph: MRR=%.2f(±%.2f)  NDCG=%.2f(±%.2f)  Recall@k=%.2f(±%.2f)%n",
                    r.graphMetrics().avgMrr(), r.graphMetrics().stdDevMrr(),
                    r.graphMetrics().avgNdcgAtK(), r.graphMetrics().stdDevNdcgAtK(),
                    r.graphMetrics().avgRecallAtK(), r.graphMetrics().stdDevRecallAtK());
            System.out.printf("  Aggregiert LLM  : Recall=%.2f(±%.2f)  Prec=%.2f(±%.2f)  F1=%.2f(±%.2f)  Hit=%.2f  MRR=%.2f(±%.2f)%n",
                    r.llmMetrics().avgRecall(), r.llmMetrics().stdDevRecall(),
                    r.llmMetrics().avgPrecision(), r.llmMetrics().stdDevPrecision(),
                    r.llmMetrics().avgF1(), r.llmMetrics().stdDevF1(),
                    r.llmMetrics().hitRate(),
                    r.llmMetrics().avgMrr(), r.llmMetrics().stdDevMrr());
        }
        System.out.println();
        System.out.println(thick);
    }
}
