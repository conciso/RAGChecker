package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.EvalResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Service
public class ReportWriter {

    private static final Logger log = LoggerFactory.getLogger(ReportWriter.class);
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private final String outputPath;
    private final String queryMode;
    private final int topK;
    private final String testCasesPath;

    private final JsonReportWriter jsonReportWriter;
    private final MarkdownReportWriter markdownReportWriter;
    private final HtmlReportWriter htmlReportWriter;

    public ReportWriter(
            @Value("${ragchecker.output.path}") String outputPath,
            @Value("${ragchecker.query.mode}") String queryMode,
            @Value("${ragchecker.query.top-k}") int topK,
            @Value("${ragchecker.testcases.path}") String testCasesPath,
            JsonReportWriter jsonReportWriter,
            MarkdownReportWriter markdownReportWriter,
            HtmlReportWriter htmlReportWriter
    ) {
        this.outputPath = outputPath;
        this.queryMode = queryMode;
        this.topK = topK;
        this.testCasesPath = testCasesPath;
        this.jsonReportWriter = jsonReportWriter;
        this.markdownReportWriter = markdownReportWriter;
        this.htmlReportWriter = htmlReportWriter;
    }

    public void write(List<EvalResult> results) {
        ReportData data = ReportData.of(results, queryMode, topK, testCasesPath);
        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String baseName = "ragcheck_" + timestamp;

        try {
            Path dir = Path.of(outputPath);
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
