package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class ReportWriter {

    private static final Logger log = LoggerFactory.getLogger(ReportWriter.class);
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private final String outputPath;
    private final String runGroup;
    private final int runsPerTestCase;
    private final String testCasesPath;
    private final String runLabelOverride;

    private final OverrideEnvLoader overrideEnvLoader;
    private final JsonReportWriter jsonReportWriter;
    private final MarkdownReportWriter markdownReportWriter;
    private final HtmlReportWriter htmlReportWriter;

    public ReportWriter(
            @Value("${ragchecker.output.path}") String outputPath,
            @Value("${ragchecker.run.group:}") String runGroup,
            @Value("${ragchecker.runs.per-testcase:1}") int runsPerTestCase,
            @Value("${ragchecker.testcases.path}") String testCasesPath,
            @Value("${ragchecker.run.label:}") String runLabelOverride,
            OverrideEnvLoader overrideEnvLoader,
            JsonReportWriter jsonReportWriter,
            MarkdownReportWriter markdownReportWriter,
            HtmlReportWriter htmlReportWriter
    ) {
        this.outputPath = outputPath;
        this.runGroup = runGroup;
        this.runsPerTestCase = runsPerTestCase;
        this.testCasesPath = testCasesPath;
        this.runLabelOverride = runLabelOverride;
        this.overrideEnvLoader = overrideEnvLoader;
        this.jsonReportWriter = jsonReportWriter;
        this.markdownReportWriter = markdownReportWriter;
        this.htmlReportWriter = htmlReportWriter;
    }

    public void write(List<AggregatedEvalResult> results) {
        write(results, List.of());
    }

    public void write(List<AggregatedEvalResult> results, List<String> failures) {
        Map<String, String> overrideParams = overrideEnvLoader.load();

        // label: RAGCHECKER_RUN_LABEL hat Vorrang, sonst aus Env-Parametern gebaut
        String runLabel = (runLabelOverride != null && !runLabelOverride.isBlank())
                ? runLabelOverride
                : overrideEnvLoader.buildLabel();

        // topK aus TOP_K-Env-Var
        int topK = 0;
        String topKStr = overrideParams.get("TOP_K");
        if (topKStr != null) {
            try { topK = Integer.parseInt(topKStr.trim()); } catch (NumberFormatException ignored) {}
        }

        // runParameters: alle Parameter außer TOP_K (wird separat als topK-Feld erfasst)
        Map<String, String> runParameters = new LinkedHashMap<>(overrideParams);
        runParameters.remove("TOP_K");

        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String baseName = (runLabel != null && !runLabel.isBlank())
                ? runLabel
                : "ragcheck_" + timestamp;

        Path base = Path.of(outputPath);
        if (runGroup != null && !runGroup.isBlank()) {
            base = base.resolve(runGroup);
        }

        // Ergebnisse nach queryMode gruppieren
        Map<String, List<AggregatedEvalResult>> byMode = new LinkedHashMap<>();
        for (AggregatedEvalResult r : results) {
            byMode.computeIfAbsent(r.queryMode(), k -> new java.util.ArrayList<>()).add(r);
        }

        for (Map.Entry<String, List<AggregatedEvalResult>> entry : byMode.entrySet()) {
            writeOneMode(entry.getValue(), entry.getKey(), baseName, runLabel,
                    runParameters, topK, failures, base);
        }
    }

    private void writeOneMode(List<AggregatedEvalResult> modeResults, String mode,
            String baseName, String runLabel, Map<String, String> runParameters,
            int topK, List<String> failures, Path base) {
        try {
            ReportData data = ReportData.of(modeResults, runLabel, runParameters,
                    List.of(mode), topK, runsPerTestCase, testCasesPath, List.of());

            // Pfad: <base>/<mode>/<baseName>/
            Path dir = base.resolve(mode).resolve(baseName);
            Files.createDirectories(dir);

            Path jsonPath = dir.resolve(baseName + ".json");
            jsonReportWriter.write(data, jsonPath);
            log.trace("JSON report written: {}", jsonPath);

            Path mdPath = dir.resolve(baseName + ".md");
            markdownReportWriter.write(data, mdPath);
            log.trace("Markdown report written: {}", mdPath);

            Path htmlPath = dir.resolve(baseName + ".html");
            htmlReportWriter.write(data, htmlPath);
            log.trace("HTML report written: {}", htmlPath);

            if (!failures.isEmpty()) {
                String labelPrefix = "[" + baseName + "/" + mode + "] ";
                StringBuilder errorLogContent = new StringBuilder();
                for (String failure : failures) {
                    errorLogContent.append(labelPrefix).append(failure).append(System.lineSeparator());
                }
                Path errorLogPath = base.resolve("errors.log");
                Files.writeString(errorLogPath, errorLogContent.toString(),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                log.warn("Errors written to: {} ({} Einträge)", errorLogPath, failures.size());
            }

        } catch (IOException e) {
            log.error("Failed to write reports for mode {}", mode, e);
        }
    }

}
