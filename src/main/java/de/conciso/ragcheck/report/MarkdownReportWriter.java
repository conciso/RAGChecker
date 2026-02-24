package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.GraphRetrievalRunResult;
import de.conciso.ragcheck.model.LlmRunResult;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

@Component
public class MarkdownReportWriter {

    private static final DateTimeFormatter DISPLAY_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    public void write(ReportData data, Path path) throws IOException {
        StringBuilder sb = new StringBuilder();
        String ts = DISPLAY_FORMAT.format(data.timestamp());

        sb.append("# RAGChecker Report — ").append(ts).append("\n\n");

        // Konfiguration
        sb.append("## Konfiguration\n\n");
        sb.append("| Parameter | Wert |\n|---|---|\n");
        if (data.runLabel() != null && !data.runLabel().isBlank()) {
            sb.append("| Run-Label | ").append(data.runLabel()).append(" |\n");
        }
        if (data.runParameters() != null && !data.runParameters().isEmpty()) {
            data.runParameters().forEach((k, v) ->
                    sb.append("| ").append(k).append(" | ").append(v).append(" |\n"));
        }
        sb.append("| Query Mode | ").append(data.queryMode()).append(" |\n");
        sb.append("| Top-K | ").append(data.topK()).append(" |\n");
        sb.append("| Läufe je Testfall | ").append(data.runsPerTestCase()).append(" |\n");
        sb.append("| Testfälle | ").append(data.testCasesPath()).append(" |\n\n");

        // Zusammenfassung
        sb.append("## Zusammenfassung\n\n");
        sb.append("### Graph-Retrieval (`/query/data`)\n\n");
        sb.append("| Metrik | Wert |\n|---|---|\n");
        sb.append(String.format("| Ø MRR | %.2f |%n", data.avgGraphMrr()));
        sb.append(String.format("| Ø NDCG@k | %.2f |%n", data.avgGraphNdcgAtK()));
        sb.append(String.format("| Ø Recall@k | %.2f |%n", data.avgGraphRecallAtK()));
        sb.append("\n");

        sb.append("### LLM (`/query`)\n\n");
        sb.append("| Metrik | Wert |\n|---|---|\n");
        sb.append(String.format("| Ø Recall | %.2f |%n", data.avgLlmRecall()));
        sb.append(String.format("| Ø Precision | %.2f |%n", data.avgLlmPrecision()));
        sb.append(String.format("| Ø F1 | %.2f |%n", data.avgLlmF1()));
        sb.append(String.format("| Ø Hit-Rate | %.2f |%n", data.avgLlmHitRate()));
        sb.append(String.format("| Ø MRR | %.2f |%n", data.avgLlmMrr()));
        sb.append("\n");

        // Ergebnis-Tabellen
        sb.append("## Ergebnisse\n\n");
        sb.append("### Graph-Retrieval\n\n");
        sb.append("| ID | Status | MRR | ±σ | NDCG@k | ±σ | Recall@k | ±σ |\n");
        sb.append("|---|---|---|---|---|---|---|---|\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append(String.format("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    r.testCaseId(), statusEmoji(r.graphMetrics().avgMrr()),
                    r.graphMetrics().avgMrr(), r.graphMetrics().stdDevMrr(),
                    r.graphMetrics().avgNdcgAtK(), r.graphMetrics().stdDevNdcgAtK(),
                    r.graphMetrics().avgRecallAtK(), r.graphMetrics().stdDevRecallAtK()));
        }
        sb.append("\n");

        sb.append("### LLM\n\n");
        sb.append("| ID | Status | Recall | ±σ | Precision | ±σ | F1 | ±σ | Hit-Rate | MRR |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|---|\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append(String.format("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    r.testCaseId(), statusEmoji(r.llmMetrics().avgF1()),
                    r.llmMetrics().avgRecall(), r.llmMetrics().stdDevRecall(),
                    r.llmMetrics().avgPrecision(), r.llmMetrics().stdDevPrecision(),
                    r.llmMetrics().avgF1(), r.llmMetrics().stdDevF1(),
                    r.llmMetrics().hitRate(), r.llmMetrics().avgMrr()));
        }
        sb.append("\n");

        // Details
        sb.append("## Details\n\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append("### ").append(r.testCaseId()).append("\n\n");
            sb.append("**Prompt:** ").append(r.prompt()).append("\n\n");
            sb.append("**Erwartet:** ").append(String.join(", ", r.expectedDocuments())).append("\n\n");

            sb.append("#### Graph-Retrieval — Ränge je erwartetem Dokument\n\n");
            sb.append("| Dokument | ");
            for (int i = 1; i <= r.runs(); i++) sb.append("Lauf ").append(i).append(" | ");
            sb.append("\n|---|");
            for (int i = 0; i < r.runs(); i++) sb.append("---|");
            sb.append("\n");
            for (Map.Entry<String, List<Integer>> e : r.graphMetrics().ranksByDoc().entrySet()) {
                sb.append("| ").append(e.getKey()).append(" | ");
                for (int rank : e.getValue()) {
                    sb.append(rank == 0 ? "—" : rank).append(" | ");
                }
                sb.append("\n");
            }
            sb.append("\n");

            sb.append("#### Graph-Retrieval — Läufe\n\n");
            sb.append("| Lauf | MRR | NDCG@k | Recall@k | Dauer (ms) | Gefunden (Top 5) |\n|---|---|---|---|---|---|\n");
            for (int i = 0; i < r.graphRuns().size(); i++) {
                GraphRetrievalRunResult g = r.graphRuns().get(i);
                String top5 = g.retrievedDocuments().stream().limit(5)
                        .reduce((a, b) -> a + ", " + b).orElse("—");
                sb.append(String.format("| %d | %.2f | %.2f | %.2f | %d | %s |%n",
                        i + 1, g.mrr(), g.ndcgAtK(), g.recallAtK(), g.durationMs(), top5));
            }
            sb.append("\n");

            sb.append("#### LLM — Läufe\n\n");
            sb.append("| Lauf | Recall | Precision | F1 | Hit | Dauer (ms) | Gefunden |\n|---|---|---|---|---|---|---|\n");
            for (int i = 0; i < r.llmRuns().size(); i++) {
                LlmRunResult l = r.llmRuns().get(i);
                sb.append(String.format("| %d | %.2f | %.2f | %.2f | %s | %d | %s |%n",
                        i + 1, l.recall(), l.precision(), l.f1(),
                        l.hit() ? "✓" : "✗",
                        l.durationMs(),
                        joinList(l.retrievedDocuments())));
            }
            sb.append("\n");
        }

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private String statusEmoji(double v) {
        if (v >= 0.8) return "✅";
        if (v >= 0.5) return "⚠️";
        return "❌";
    }

    private String joinList(List<String> list) {
        if (list == null || list.isEmpty()) return "—";
        return String.join(", ", list);
    }
}
