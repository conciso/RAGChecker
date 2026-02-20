package de.conciso.ragcheck.report;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import de.conciso.ragcheck.model.EvalResult;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class JsonReportWriter {

    private final ObjectMapper objectMapper;

    public JsonReportWriter() {
        this.objectMapper = new ObjectMapper();
        this.objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void write(ReportData data, Path path) throws IOException {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("timestamp", data.timestamp().toString());

        Map<String, Object> config = new LinkedHashMap<>();
        config.put("queryMode", data.queryMode());
        config.put("topK", data.topK());
        config.put("testCasesPath", data.testCasesPath());
        json.put("configuration", config);

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("avgRecall", round(data.avgRecall()));
        summary.put("avgPrecision", round(data.avgPrecision()));
        summary.put("avgF1", round(data.avgF1()));
        summary.put("totalTestCases", data.results().size());
        json.put("summary", summary);

        List<Map<String, Object>> results = data.results().stream().map(r -> {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("id", r.testCaseId());
            entry.put("prompt", r.prompt());
            entry.put("recall", round(r.recall()));
            entry.put("precision", round(r.precision()));
            entry.put("f1", round(r.f1()));
            entry.put("expectedDocuments", r.expectedDocuments());
            entry.put("retrievedDocuments", r.retrievedDocuments());
            if (r.queryResult() != null) {
                entry.put("entityCount", r.queryResult().entityCount());
                entry.put("relationshipCount", r.queryResult().relationshipCount());
                entry.put("chunkCount", r.queryResult().chunkCount());
                entry.put("highLevelKeywords", r.queryResult().highLevelKeywords());
                entry.put("lowLevelKeywords", r.queryResult().lowLevelKeywords());
            }
            return entry;
        }).toList();

        json.put("results", results);

        objectMapper.writeValue(path.toFile(), json);
    }

    private double round(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
