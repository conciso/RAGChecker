package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
    private final List<String> queryModes;
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
            @Value("${ragchecker.query.modes}") String queryModesStr,
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
        this.queryModes = List.of(queryModesStr.split(","));
        this.runsPerTestCase = runsPerTestCase;
        this.testCasesPath = testCasesPath;
        this.runLabelOverride = runLabelOverride;
        this.overrideEnvLoader = overrideEnvLoader;
        this.jsonReportWriter = jsonReportWriter;
        this.markdownReportWriter = markdownReportWriter;
        this.htmlReportWriter = htmlReportWriter;
    }

    public void write(List<AggregatedEvalResult> results) {
        Map<String, String> overrideParams = overrideEnvLoader.load();

        // label: RAGCHECKER_RUN_LABEL hat Vorrang, sonst "label" aus override.env
        String runLabel = (runLabelOverride != null && !runLabelOverride.isBlank())
                ? runLabelOverride
                : overrideParams.getOrDefault("label", "");

        // topK aus override.env (für den Report-Header), fallback 0
        int topK = 0;
        String topKStr = overrideParams.get("top_k");
        if (topKStr != null) {
            try { topK = Integer.parseInt(topKStr.trim()); } catch (NumberFormatException ignored) {}
        }

        // runParameters: alle Werte aus override.env außer "label" und "top_k"
        Map<String, String> runParameters = new LinkedHashMap<>(overrideParams);
        runParameters.remove("label");
        runParameters.remove("top_k");

        ReportData data = ReportData.of(results, runLabel, runParameters,
                queryModes, topK, runsPerTestCase, testCasesPath);

        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String baseName = (runLabel != null && !runLabel.isBlank())
                ? runLabel
                : "ragcheck_" + timestamp;

        try {
            // Dateien kommen in <outputPath>/[<runGroup>/]<baseName>/
            Path base = Path.of(outputPath);
            if (runGroup != null && !runGroup.isBlank()) {
                base = base.resolve(runGroup);
            }
            Path dir = base.resolve(baseName);
            Files.createDirectories(dir);

            Path jsonPath = dir.resolve(baseName + ".json");
            jsonReportWriter.write(data, jsonPath);
            log.info("JSON report written: {}", jsonPath);

            Path mdPath = dir.resolve(baseName + ".md");
            markdownReportWriter.write(data, mdPath);
            log.info("Markdown report written: {}", mdPath);

            Path htmlPath = dir.resolve(baseName + ".html");
            htmlReportWriter.write(data, htmlPath);
            log.info("HTML report written: {}", htmlPath);

        } catch (IOException e) {
            log.error("Failed to write reports", e);
        }
    }

}
