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
import java.util.LinkedHashMap;
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
    private final String runGroup;
    private final ComparisonReportWriter comparisonReportWriter;
    private final ObjectMapper objectMapper;

    public ComparisonRunner(
            @Value("${ragchecker.output.path}") String outputPath,
            @Value("${ragchecker.run.group:}") String runGroup,
            ComparisonReportWriter comparisonReportWriter
    ) {
        this.outputPath = outputPath;
        this.runGroup = runGroup;
        this.comparisonReportWriter = comparisonReportWriter;
        this.objectMapper = new ObjectMapper();
    }

    @Override
    public void run(String... args) throws Exception {
        Path base = Path.of(outputPath);
        if (runGroup != null && !runGroup.isBlank()) {
            base = base.resolve(runGroup);
        }
        Path dir = base;
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

        List<ComparisonEntry> allEntries = new ArrayList<>();
        for (Path file : reportFiles) {
            try {
                ComparisonEntry entry = parseReport(file);
                allEntries.add(entry);
                log.info("  Gelesen: {} (Label: {}, Mode: {})", file.getFileName(),
                        entry.runLabel().isBlank() ? "—" : entry.runLabel(),
                        entry.queryMode());
            } catch (Exception e) {
                log.warn("  Fehler beim Lesen von {}: {}", file.getFileName(), e.getMessage());
            }
        }

        if (allEntries.isEmpty()) {
            log.error("Keine gültigen Reports gelesen, Vergleich abgebrochen.");
            return;
        }

        // Entries nach Mode gruppieren und pro Mode einen Vergleichsreport schreiben
        Map<String, List<ComparisonEntry>> byMode = new LinkedHashMap<>();
        for (ComparisonEntry entry : allEntries) {
            byMode.computeIfAbsent(entry.queryMode(), k -> new ArrayList<>()).add(entry);
        }

        for (Map.Entry<String, List<ComparisonEntry>> modeEntry : byMode.entrySet()) {
            String mode = modeEntry.getKey();
            List<ComparisonEntry> modeEntries = modeEntry.getValue();
            Path modeDir = dir.resolve(mode);
            if (Files.isDirectory(modeDir)) {
                comparisonReportWriter.write(modeEntries, modeDir);
                log.info("Mode-Vergleichsreport geschrieben: {}/comparison.{{md,html}}", modeDir.toAbsolutePath());
            } else {
                log.warn("Mode-Verzeichnis nicht gefunden, überspringe per-Mode-Report: {}", modeDir);
            }
        }

        // Mode-übergreifender Vergleich (nur wenn mehr als 1 Mode vorhanden)
        if (byMode.size() > 1) {
            comparisonReportWriter.write(allEntries, dir);
            log.info("Vergleichsreport (alle Modi) geschrieben: {}/comparison.{{md,html}}", dir.toAbsolutePath());
        } else {
            // Nur ein Mode: Gesamt-Report trotzdem schreiben (entspricht dem Mode-Report)
            comparisonReportWriter.write(allEntries, dir);
            log.info("Vergleichsreport geschrieben: {}/comparison.{{md,html}}", dir.toAbsolutePath());
        }

        printSummary(allEntries);
    }

    // -------------------------------------------------------------------------

    private List<Path> findReportFiles(Path dir) throws IOException {
        List<Path> files = new ArrayList<>();
        try (var modeStream = Files.list(dir)) {
            for (Path modeDir : modeStream.filter(Files::isDirectory).sorted().toList()) {
                // Neue Struktur: <modeDir>/<label>/<label>.json
                try (var labelStream = Files.list(modeDir)) {
                    labelStream.filter(Files::isDirectory).sorted()
                            .map(labelDir -> labelDir.resolve(labelDir.getFileName() + ".json"))
                            .filter(Files::isRegularFile)
                            .forEach(files::add);
                } catch (IOException e) {
                    log.warn("Fehler beim Lesen von Mode-Verzeichnis {}: {}", modeDir, e.getMessage());
                }
                // Legacy-Fallback: <modeDir> ist selbst ein Label-Ordner
                Path legacyFile = modeDir.resolve(modeDir.getFileName() + ".json");
                if (Files.isRegularFile(legacyFile) && !files.contains(legacyFile)) {
                    files.add(legacyFile);
                }
            }
        }
        return files;
    }

    @SuppressWarnings("unchecked")
    private ComparisonEntry parseReport(Path file) throws IOException {
        Map<String, Object> root = objectMapper.readValue(file.toFile(),
                new TypeReference<Map<String, Object>>() {});

        String timestamp = (String) root.getOrDefault("timestamp", "");

        Map<String, Object> config = (Map<String, Object>) root.getOrDefault("configuration", Map.of());
        String runLabel      = (String) config.getOrDefault("runLabel", "");
        Map<String, String> runParameters = parseRunParameters(config.get("runParameters"));
        String queryMode     = parseQueryModes(config);
        int topK             = toInt(config.getOrDefault("topK", 0));
        int runsPerTestCase  = toInt(config.getOrDefault("runsPerTestCase", 1));

        Map<String, Object> summary = (Map<String, Object>) root.getOrDefault("summary", Map.of());
        Map<String, Object> graph   = (Map<String, Object>) summary.getOrDefault("graph", Map.of());
        Map<String, Object> llm     = (Map<String, Object>) summary.getOrDefault("llm", Map.of());
        int totalTestCases           = toInt(summary.getOrDefault("totalTestCases", 0));
        double avgGraphDurationSec   = toDouble(graph.getOrDefault("avgDurationSec", 0.0));
        double avgLlmDurationSec     = toDouble(llm.getOrDefault("avgDurationSec", 0.0));

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
                tcLlmRecall, tcLlmPrecision, tcLlmF1, tcLlmHitRate, tcLlmMrr,
                totalTestCases, avgGraphDurationSec, avgLlmDurationSec
        );
    }

    @SuppressWarnings("unchecked")
    private String parseQueryModes(Map<String, Object> config) {
        Object raw = config.get("queryModes");
        if (raw instanceof List<?> list) {
            return list.stream().map(Object::toString).collect(java.util.stream.Collectors.joining(", "));
        }
        // Fallback: altes Format mit "queryMode" (einzelner String)
        Object legacy = config.get("queryMode");
        return legacy instanceof String s ? s : "";
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
