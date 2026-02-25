package de.conciso.ragcheck.runner;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.conciso.ragcheck.report.ComparisonReportWriter;
import de.conciso.ragcheck.report.ComparisonReportWriter.ComparisonEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Wird aktiv wenn RAGCHECKER_MODE=compare.
 * Liest alle <subdir>/<subdir>.json Unterverzeichnisse im Output-Verzeichnis
 * und erstellt einen Vergleichsreport (compare.json/.md/.html).
 */
@Component
@ConditionalOnProperty(name = "ragchecker.mode", havingValue = "compare")
public class ComparisonRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(ComparisonRunner.class);

    private final String outputPath;
    private final ComparisonReportWriter comparisonReportWriter;
    private final ObjectMapper objectMapper;

    public ComparisonRunner(
            @Value("${ragchecker.output.path}") String outputPath,
            ComparisonReportWriter comparisonReportWriter
    ) {
        this.outputPath = outputPath;
        this.comparisonReportWriter = comparisonReportWriter;
        this.objectMapper = new ObjectMapper();
    }

    @Override
    public void run(String... args) throws Exception {
        Path dir = Path.of(outputPath);
        if (!Files.isDirectory(dir)) {
            log.error("Output-Verzeichnis nicht gefunden: {}", dir.toAbsolutePath());
            return;
        }

        List<Path> reportFiles = findReportFiles(dir);
        if (reportFiles.isEmpty()) {
            log.warn("Keine Report-Unterverzeichnisse in {} gefunden.", dir.toAbsolutePath());
            return;
        }

        log.info("{} Report-Datei(en) gefunden, lese Daten...", reportFiles.size());

        List<ComparisonEntry> entries = new ArrayList<>();
        for (Path file : reportFiles) {
            try {
                ComparisonEntry entry = parseReport(file);
                entries.add(entry);
                log.info("  Gelesen: {} (Label: {})", file.getFileName(),
                        entry.runLabel().isBlank() ? "—" : entry.runLabel());
            } catch (Exception e) {
                log.warn("  Fehler beim Lesen von {}: {}", file.getFileName(), e.getMessage());
            }
        }

        if (entries.isEmpty()) {
            log.error("Keine gültigen Reports gelesen, Vergleich abgebrochen.");
            return;
        }

        comparisonReportWriter.write(entries, dir);
        log.info("Vergleichsreport geschrieben: {}/comparison.{{md,html}}", dir.toAbsolutePath());

        printSummary(entries);
    }

    // -------------------------------------------------------------------------

    private List<Path> findReportFiles(Path dir) throws IOException {
        try (var stream = Files.list(dir)) {
            return stream
                    .filter(Files::isDirectory)
                    .map(subDir -> subDir.resolve(subDir.getFileName() + ".json"))
                    .filter(Files::isRegularFile)
                    .sorted()
                    .toList();
        }
    }

    @SuppressWarnings("unchecked")
    private ComparisonEntry parseReport(Path file) throws IOException {
        Map<String, Object> root = objectMapper.readValue(file.toFile(),
                new TypeReference<Map<String, Object>>() {});

        String timestamp = (String) root.getOrDefault("timestamp", "");

        Map<String, Object> config = (Map<String, Object>) root.getOrDefault("configuration", Map.of());
        String runLabel      = (String) config.getOrDefault("runLabel", "");
        Map<String, String> runParameters = parseRunParameters(config.get("runParameters"));
        String queryMode     = (String) config.getOrDefault("queryMode", "");
        int topK             = toInt(config.getOrDefault("topK", 0));
        int runsPerTestCase  = toInt(config.getOrDefault("runsPerTestCase", 1));

        Map<String, Object> summary = (Map<String, Object>) root.getOrDefault("summary", Map.of());
        Map<String, Object> graph   = (Map<String, Object>) summary.getOrDefault("graph", Map.of());
        Map<String, Object> llm     = (Map<String, Object>) summary.getOrDefault("llm", Map.of());

        // Per-Testfall-Daten für Boxplots
        List<Object> results = (List<Object>) root.getOrDefault("results", List.of());
        List<Double> tcGraphMrr     = new ArrayList<>();
        List<Double> tcGraphNdcg    = new ArrayList<>();
        List<Double> tcGraphRecall  = new ArrayList<>();
        List<Double> tcLlmRecall    = new ArrayList<>();
        List<Double> tcLlmPrecision = new ArrayList<>();
        List<Double> tcLlmF1        = new ArrayList<>();
        List<Double> tcLlmHitRate   = new ArrayList<>();
        List<Double> tcLlmMrr       = new ArrayList<>();

        for (Object r : results) {
            Map<String, Object> resultMap = (Map<String, Object>) r;
            Map<String, Object> g = (Map<String, Object>) resultMap.getOrDefault("graph", Map.of());
            Map<String, Object> l = (Map<String, Object>) resultMap.getOrDefault("llm", Map.of());
            tcGraphMrr.add(toDouble(g.getOrDefault("avgMrr", 0.0)));
            tcGraphNdcg.add(toDouble(g.getOrDefault("avgNdcgAtK", 0.0)));
            tcGraphRecall.add(toDouble(g.getOrDefault("avgRecallAtK", 0.0)));
            tcLlmRecall.add(toDouble(l.getOrDefault("avgRecall", 0.0)));
            tcLlmPrecision.add(toDouble(l.getOrDefault("avgPrecision", 0.0)));
            tcLlmF1.add(toDouble(l.getOrDefault("avgF1", 0.0)));
            tcLlmHitRate.add(toDouble(l.getOrDefault("hitRate", 0.0)));
            tcLlmMrr.add(toDouble(l.getOrDefault("avgMrr", 0.0)));
        }

        return new ComparisonEntry(
                timestamp,
                runLabel != null ? runLabel : "",
                runParameters,
                queryMode != null ? queryMode : "",
                topK,
                runsPerTestCase,
                toDouble(graph.getOrDefault("avgMrr", 0.0)),
                toDouble(graph.getOrDefault("avgNdcgAtK", 0.0)),
                toDouble(graph.getOrDefault("avgRecallAtK", 0.0)),
                toDouble(llm.getOrDefault("avgRecall", 0.0)),
                toDouble(llm.getOrDefault("avgPrecision", 0.0)),
                toDouble(llm.getOrDefault("avgF1", 0.0)),
                toDouble(llm.getOrDefault("avgHitRate", 0.0)),
                toDouble(llm.getOrDefault("avgMrr", 0.0)),
                tcGraphMrr, tcGraphNdcg, tcGraphRecall,
                tcLlmRecall, tcLlmPrecision, tcLlmF1, tcLlmHitRate, tcLlmMrr
        );
    }

    @SuppressWarnings("unchecked")
    private Map<String, String> parseRunParameters(Object raw) {
        if (raw instanceof Map<?, ?> map) {
            Map<String, String> result = new java.util.LinkedHashMap<>();
            map.forEach((k, v) -> result.put(String.valueOf(k), String.valueOf(v)));
            return result;
        }
        return Collections.emptyMap();
    }

    private double toDouble(Object v) {
        if (v instanceof Number n) return n.doubleValue();
        return 0.0;
    }

    private int toInt(Object v) {
        if (v instanceof Number n) return n.intValue();
        return 0;
    }

    private void printSummary(List<ComparisonEntry> entries) {
        List<ComparisonEntry> sorted = entries.stream()
                .sorted((a, b) -> Double.compare(b.avgLlmF1(), a.avgLlmF1()))
                .toList();

        String sep = "-".repeat(110);
        System.out.println();
        System.out.println("=== RAGChecker Parametervergleich ===");
        System.out.println("Sortiert nach Ø LLM-F1 (absteigend)\n");
        System.out.printf("%-3s %-25s %-20s %-8s %6s %8s %8s %8s %8s %8s %8s%n",
                "#", "Timestamp", "Label", "Mode", "TopK",
                "Graph-MRR", "NDCG@k", "Recall@k", "LLM-F1", "HitRate", "LLM-MRR");
        System.out.println(sep);
        for (int i = 0; i < sorted.size(); i++) {
            ComparisonEntry e = sorted.get(i);
            System.out.printf("%-3d %-25s %-20s %-8s %6d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f%n",
                    i + 1,
                    e.timestamp().length() > 24 ? e.timestamp().substring(0, 24) : e.timestamp(),
                    e.runLabel().isBlank() ? "—" : e.runLabel(),
                    e.queryMode(),
                    e.topK(),
                    e.avgGraphMrr(),
                    e.avgGraphNdcgAtK(),
                    e.avgGraphRecallAtK(),
                    e.avgLlmF1(),
                    e.avgLlmHitRate(),
                    e.avgLlmMrr());
            if (!e.runParameters().isEmpty()) {
                e.runParameters().forEach((k, v) ->
                        System.out.printf("    %s=%s%n", k, v));
            }
        }
        System.out.println(sep);
        System.out.println();
        System.out.println("Detailierter Report: " + Path.of(outputPath).toAbsolutePath()
                + "/comparison.{md,html}");
        System.out.println();
    }
}
