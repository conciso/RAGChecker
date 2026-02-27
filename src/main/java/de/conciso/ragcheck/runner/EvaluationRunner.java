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
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
@ConditionalOnProperty(name = "ragchecker.mode", havingValue = "evaluate", matchIfMissing = true)
public class EvaluationRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(EvaluationRunner.class);

    private final TestCaseLoader loader;
    private final EvaluationService evaluationService;
    private final ReportWriter reportWriter;
    private final de.conciso.ragcheck.report.OverrideEnvLoader overrideEnvLoader;
    private final String runLabelOverride;
    private final String runGroup;

    public EvaluationRunner(TestCaseLoader loader, EvaluationService evaluationService,
                            ReportWriter reportWriter,
                            de.conciso.ragcheck.report.OverrideEnvLoader overrideEnvLoader,
                            @org.springframework.beans.factory.annotation.Value("${ragchecker.run.label:}") String runLabelOverride,
                            @org.springframework.beans.factory.annotation.Value("${ragchecker.run.group:}") String runGroup) {
        this.loader = loader;
        this.evaluationService = evaluationService;
        this.reportWriter = reportWriter;
        this.overrideEnvLoader = overrideEnvLoader;
        this.runLabelOverride = runLabelOverride;
        this.runGroup = runGroup;
    }

    @Override
    public void run(String... args) throws Exception {
        String effectiveLabel = resolveLabel();
        System.out.println();
        System.out.println("=== RAGChecker ===");
        if (runGroup != null && !runGroup.isBlank()) {
            System.out.println("Run-Group : " + runGroup);
        }
        System.out.println("Run-Label : " + (effectiveLabel.isBlank() ? "(kein Label)" : effectiveLabel));
        System.out.println();

        List<TestCase> testCases = loader.load();
        log.info("Loaded {} test case(s)", testCases.size());

        List<AggregatedEvalResult> results = evaluationService.evaluate(testCases);

        printSummary(results);
        printDebug(results);
        reportWriter.write(results);
    }

    private String resolveLabel() {
        if (runLabelOverride != null && !runLabelOverride.isBlank()) return runLabelOverride;
        return overrideEnvLoader.load().getOrDefault("label", "");
    }

    // -------------------------------------------------------------------------

    private record ModeStat(
            String mode,
            double graphMrr, double graphNdcg, double graphRecall,
            double llmRecall, double llmPrec, double llmF1, double llmHit, double llmMrr
    ) {}

    private void printSummary(List<AggregatedEvalResult> results) {
        // Group by mode, preserving order of first occurrence
        Map<String, List<AggregatedEvalResult>> byMode = new LinkedHashMap<>();
        for (AggregatedEvalResult r : results) {
            byMode.computeIfAbsent(r.queryMode(), k -> new ArrayList<>()).add(r);
        }

        List<ModeStat> stats = byMode.entrySet().stream().map(e -> {
            var rs = e.getValue();
            return new ModeStat(e.getKey(),
                    rs.stream().mapToDouble(r -> r.graphMetrics().avgMrr()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.graphMetrics().avgNdcgAtK()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.graphMetrics().avgRecallAtK()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.llmMetrics().avgRecall()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.llmMetrics().avgPrecision()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.llmMetrics().avgF1()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.llmMetrics().hitRate()).average().orElse(0),
                    rs.stream().mapToDouble(r -> r.llmMetrics().avgMrr()).average().orElse(0));
        }).toList();

        long tcCount = results.stream().map(AggregatedEvalResult::testCaseId).distinct().count();
        int runsPerMode = byMode.values().stream().findFirst().map(List::size).orElse(0);
        String sep = "-".repeat(72);

        System.out.println();
        System.out.printf("=== RAGChecker Summary — %d Testfall/Testfälle × %d Modi × %d Läufe ===%n",
                tcCount, byMode.size(), runsPerMode > 0 ? results.get(0).runs() : 0);
        System.out.println();

        // --- Graph ---
        double bestGraphRecall = stats.stream().mapToDouble(ModeStat::graphRecall).max().orElse(0);
        System.out.println("-- Graph-Retrieval (/query/data) --");
        System.out.printf("%-10s %8s %10s %12s%n", "Mode", "Ø MRR", "Ø NDCG@k", "Ø Recall@k");
        System.out.println(sep);
        for (ModeStat s : stats) {
            boolean best = Math.abs(s.graphRecall() - bestGraphRecall) < 0.001;
            System.out.printf("%-10s %8.2f %10.2f %12.2f%s%n",
                    s.mode(), s.graphMrr(), s.graphNdcg(), s.graphRecall(),
                    best ? "  ★" : "");
        }

        // --- LLM ---
        double bestLlmF1 = stats.stream().mapToDouble(ModeStat::llmF1).max().orElse(0);
        System.out.println();
        System.out.println("-- LLM (/query) --");
        System.out.printf("%-10s %8s %11s %8s %8s %8s%n",
                "Mode", "Ø Recall", "Ø Precision", "Ø F1", "Hit%", "Ø MRR");
        System.out.println(sep);
        for (ModeStat s : stats) {
            boolean best = Math.abs(s.llmF1() - bestLlmF1) < 0.001;
            System.out.printf("%-10s %8.2f %11.2f %8.2f %8.2f %8.2f%s%n",
                    s.mode(), s.llmRecall(), s.llmPrec(), s.llmF1(),
                    s.llmHit(), s.llmMrr(),
                    best ? "  ★ bester Mode" : "");
        }
        System.out.println(sep);
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
                System.out.printf("    MRR=%.2f  NDCG@k=%.2f  Recall@k=%.2f  Dauer=%.2fs%n",
                        g.mrr(), g.ndcgAtK(), g.recallAtK(), g.durationMs() / 1000.0);
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
                System.out.printf("    Recall=%.2f  Precision=%.2f  F1=%.2f  Hit=%s  MRR=%.2f  Dauer=%.2fs%n",
                        l.recall(), l.precision(), l.f1(), l.hit() ? "✓" : "✗", l.mrr(), l.durationMs() / 1000.0);
                System.out.println("    Gefunden: " +
                        (l.retrievedDocuments().isEmpty() ? "(keine)" : String.join(", ", l.retrievedDocuments())));
                System.out.println(thin);
            }

            System.out.printf("  Aggregiert Graph: MRR=%.2f(±%.2f)  NDCG=%.2f(±%.2f)  Recall@k=%.2f(±%.2f)  ØDauer=%.2fs%n",
                    r.graphMetrics().avgMrr(), r.graphMetrics().stdDevMrr(),
                    r.graphMetrics().avgNdcgAtK(), r.graphMetrics().stdDevNdcgAtK(),
                    r.graphMetrics().avgRecallAtK(), r.graphMetrics().stdDevRecallAtK(),
                    r.graphMetrics().avgDurationMs() / 1000.0);
            System.out.printf("  Aggregiert LLM  : Recall=%.2f(±%.2f)  Prec=%.2f(±%.2f)  F1=%.2f(±%.2f)  Hit=%.2f  MRR=%.2f(±%.2f)  ØDauer=%.2fs%n",
                    r.llmMetrics().avgRecall(), r.llmMetrics().stdDevRecall(),
                    r.llmMetrics().avgPrecision(), r.llmMetrics().stdDevPrecision(),
                    r.llmMetrics().avgF1(), r.llmMetrics().stdDevF1(),
                    r.llmMetrics().hitRate(),
                    r.llmMetrics().avgMrr(), r.llmMetrics().stdDevMrr(),
                    r.llmMetrics().avgDurationMs() / 1000.0);
        }
        System.out.println();
        System.out.println(thick);
    }
}
