package de.conciso.ragcheck.report;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.GraphRetrievalRunResult;
import de.conciso.ragcheck.model.LlmRunResult;
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
        Map<String, Object> graphSummary = new LinkedHashMap<>();
        graphSummary.put("avgMrr", r(data.avgGraphMrr()));
        graphSummary.put("avgNdcgAtK", r(data.avgGraphNdcgAtK()));
        graphSummary.put("avgRecallAtK", r(data.avgGraphRecallAtK()));
        summary.put("graph", graphSummary);

        Map<String, Object> llmSummary = new LinkedHashMap<>();
        llmSummary.put("avgRecall", r(data.avgLlmRecall()));
        llmSummary.put("avgPrecision", r(data.avgLlmPrecision()));
        llmSummary.put("avgF1", r(data.avgLlmF1()));
        llmSummary.put("avgHitRate", r(data.avgLlmHitRate()));
        llmSummary.put("avgMrr", r(data.avgLlmMrr()));
        summary.put("llm", llmSummary);
        summary.put("totalTestCases", data.results().size());
        json.put("summary", summary);

        List<Map<String, Object>> results = data.results().stream().map(this::resultToMap).toList();
        json.put("results", results);

        objectMapper.writeValue(path.toFile(), json);
    }

    private Map<String, Object> resultToMap(AggregatedEvalResult r) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", r.testCaseId());
        entry.put("prompt", r.prompt());
        entry.put("expectedDocuments", r.expectedDocuments());
        entry.put("runs", r.runs());

        Map<String, Object> graph = new LinkedHashMap<>();
        graph.put("avgMrr", r(r.graphMetrics().avgMrr()));
        graph.put("stdDevMrr", r(r.graphMetrics().stdDevMrr()));
        graph.put("avgNdcgAtK", r(r.graphMetrics().avgNdcgAtK()));
        graph.put("stdDevNdcgAtK", r(r.graphMetrics().stdDevNdcgAtK()));
        graph.put("avgRecallAtK", r(r.graphMetrics().avgRecallAtK()));
        graph.put("stdDevRecallAtK", r(r.graphMetrics().stdDevRecallAtK()));
        graph.put("ranksByDoc", r.graphMetrics().ranksByDoc());
        graph.put("runs", r.graphRuns().stream().map(this::graphRunToMap).toList());
        entry.put("graph", graph);

        Map<String, Object> llm = new LinkedHashMap<>();
        llm.put("avgRecall", r(r.llmMetrics().avgRecall()));
        llm.put("stdDevRecall", r(r.llmMetrics().stdDevRecall()));
        llm.put("avgPrecision", r(r.llmMetrics().avgPrecision()));
        llm.put("stdDevPrecision", r(r.llmMetrics().stdDevPrecision()));
        llm.put("avgF1", r(r.llmMetrics().avgF1()));
        llm.put("stdDevF1", r(r.llmMetrics().stdDevF1()));
        llm.put("hitRate", r(r.llmMetrics().hitRate()));
        llm.put("avgMrr", r(r.llmMetrics().avgMrr()));
        llm.put("stdDevMrr", r(r.llmMetrics().stdDevMrr()));
        llm.put("runs", r.llmRuns().stream().map(this::llmRunToMap).toList());
        entry.put("llm", llm);

        return entry;
    }

    private Map<String, Object> graphRunToMap(GraphRetrievalRunResult run) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("retrievedDocuments", run.retrievedDocuments());
        m.put("expectedDocumentRanks", run.expectedDocumentRanks());
        m.put("mrr", r(run.mrr()));
        m.put("ndcgAtK", r(run.ndcgAtK()));
        m.put("recallAtK", r(run.recallAtK()));
        return m;
    }

    private Map<String, Object> llmRunToMap(LlmRunResult run) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("retrievedDocuments", run.retrievedDocuments());
        m.put("recall", r(run.recall()));
        m.put("precision", r(run.precision()));
        m.put("f1", r(run.f1()));
        m.put("hit", run.hit());
        m.put("mrr", r(run.mrr()));
        m.put("responseText", run.responseText());
        return m;
    }

    private double r(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
