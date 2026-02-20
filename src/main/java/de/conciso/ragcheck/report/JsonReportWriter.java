package de.conciso.ragcheck.report;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import de.conciso.ragcheck.model.AggregatedEvalResult;
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
        config.put("runsPerTestCase", data.runsPerTestCase());
        config.put("testCasesPath", data.testCasesPath());
        json.put("configuration", config);

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("avgRecall", round(data.avgRecall()));
        summary.put("avgPrecision", round(data.avgPrecision()));
        summary.put("avgF1", round(data.avgF1()));
        summary.put("avgHitRate", round(data.avgHitRate()));
        summary.put("avgMrr", round(data.avgMrr()));
        summary.put("totalTestCases", data.results().size());
        json.put("summary", summary);

        List<Map<String, Object>> results = data.results().stream().map(r -> {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("id", r.testCaseId());
            entry.put("prompt", r.prompt());
            entry.put("runs", r.runs());
            entry.put("avgRecall", round(r.avgRecall()));
            entry.put("avgPrecision", round(r.avgPrecision()));
            entry.put("avgF1", round(r.avgF1()));
            entry.put("stdDevRecall", round(r.stdDevRecall()));
            entry.put("stdDevPrecision", round(r.stdDevPrecision()));
            entry.put("stdDevF1", round(r.stdDevF1()));
            entry.put("hitRate", round(r.hitRate()));
            entry.put("mrr", round(r.mrr()));
            entry.put("expectedDocuments", r.expectedDocuments());
            entry.put("runResults", r.runResults().stream().map(this::runToMap).toList());
            return entry;
        }).toList();

        json.put("results", results);

        objectMapper.writeValue(path.toFile(), json);
    }

    private Map<String, Object> runToMap(EvalResult r) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("recall", round(r.recall()));
        m.put("precision", round(r.precision()));
        m.put("f1", round(r.f1()));
        m.put("retrievedDocuments", r.retrievedDocuments());
        if (r.queryResult() != null) {
            m.put("entityCount", r.queryResult().entityCount());
            m.put("relationshipCount", r.queryResult().relationshipCount());
            m.put("chunkCount", r.queryResult().chunkCount());
            m.put("highLevelKeywords", r.queryResult().highLevelKeywords());
            m.put("lowLevelKeywords", r.queryResult().lowLevelKeywords());
        }
        return m;
    }

    private double round(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
