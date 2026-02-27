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
import java.util.ArrayList;
import java.util.LinkedHashMap;
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
        sb.append("| Query Modes | ").append(String.join(", ", data.queryModes())).append(" |\n");
        sb.append("| Top-K | ").append(data.topK()).append(" |\n");
        sb.append("| Läufe je Testfall | ").append(data.runsPerTestCase()).append(" |\n");
        sb.append("| Testfälle | ").append(data.testCasesPath()).append(" |\n\n");

        // Zusammenfassung
        sb.append("## Zusammenfassung\n\n");
        sb.append("### Graph-Retrieval (`/query/data`)\n\n");
        sb.append("| Metrik | Ø | ±σ |\n|---|---|---|\n");
        sb.append(String.format("| Ø MRR | %.2f | %.2f |%n", data.avgGraphMrr(), data.stdDevGraphMrr()));
        sb.append(String.format("| Ø NDCG@k | %.2f | %.2f |%n", data.avgGraphNdcgAtK(), data.stdDevGraphNdcgAtK()));
        sb.append(String.format("| Ø Recall@k | %.2f | %.2f |%n", data.avgGraphRecallAtK(), data.stdDevGraphRecallAtK()));
        sb.append("\n");

        sb.append("### LLM (`/query`)\n\n");
        sb.append("| Metrik | Ø | ±σ |\n|---|---|---|\n");
        sb.append(String.format("| Ø Recall | %.2f | %.2f |%n", data.avgLlmRecall(), data.stdDevLlmRecall()));
        sb.append(String.format("| Ø Precision | %.2f | %.2f |%n", data.avgLlmPrecision(), data.stdDevLlmPrecision()));
        sb.append(String.format("| Ø F1 | %.2f | %.2f |%n", data.avgLlmF1(), data.stdDevLlmF1()));
        sb.append(String.format("| Ø Hit-Rate | %.2f | %.2f |%n", data.avgLlmHitRate(), data.stdDevLlmHitRate()));
        sb.append(String.format("| Ø MRR | %.2f | %.2f |%n", data.avgLlmMrr(), data.stdDevLlmMrr()));
        sb.append("\n");

        // Modus-Vergleich
        sb.append("## Modus-Vergleich\n\n");
        appendModeComparison(sb, data.results());
        sb.append("\n");

        // Ergebnis-Tabellen
        sb.append("## Ergebnisse\n\n");
        sb.append("### Graph-Retrieval\n\n");
        sb.append("| Mode | ID | Status | MRR | ±σ | NDCG@k | ±σ | Recall@k | ±σ |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append(String.format("| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    r.queryMode(), r.testCaseId(), statusEmoji(r.graphMetrics().avgMrr()),
                    r.graphMetrics().avgMrr(), r.graphMetrics().stdDevMrr(),
                    r.graphMetrics().avgNdcgAtK(), r.graphMetrics().stdDevNdcgAtK(),
                    r.graphMetrics().avgRecallAtK(), r.graphMetrics().stdDevRecallAtK()));
        }
        sb.append("\n");

        sb.append("### LLM\n\n");
        sb.append("| Mode | ID | Status | Recall | ±σ | Precision | ±σ | F1 | ±σ | Hit-Rate | MRR |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append(String.format("| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    r.queryMode(), r.testCaseId(), statusEmoji(r.llmMetrics().avgF1()),
                    r.llmMetrics().avgRecall(), r.llmMetrics().stdDevRecall(),
                    r.llmMetrics().avgPrecision(), r.llmMetrics().stdDevPrecision(),
                    r.llmMetrics().avgF1(), r.llmMetrics().stdDevF1(),
                    r.llmMetrics().hitRate(), r.llmMetrics().avgMrr()));
        }
        sb.append("\n");

        // Details
        sb.append("## Details\n\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append("### ").append(r.testCaseId()).append(" / ").append(r.queryMode()).append("\n\n");
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
            sb.append("| Lauf | MRR | NDCG@k | Recall@k | Dauer (s) | Gefunden (alle) |\n|---|---|---|---|---|---|\n");
            for (int i = 0; i < r.graphRuns().size(); i++) {
                GraphRetrievalRunResult g = r.graphRuns().get(i);
                String allDocs = g.retrievedDocuments().isEmpty() ? "—"
                        : String.join(", ", g.retrievedDocuments());
                sb.append(String.format("| %d | %.2f | %.2f | %.2f | %.2f | %s |%n",
                        i + 1, g.mrr(), g.ndcgAtK(), g.recallAtK(), g.durationMs() / 1000.0, allDocs));
            }
            sb.append("\n");

            // Semantische Interpretation (nur beim ersten Lauf, da pro TC × Mode konstant)
            if (!r.graphRuns().isEmpty()) {
                GraphRetrievalRunResult first = r.graphRuns().get(0);
                boolean hasEntities = !first.entityNames().isEmpty();
                boolean hasPairs    = !first.relationshipPairs().isEmpty();
                boolean hasSources  = first.documentsBySource().values().stream().anyMatch(l -> !l.isEmpty());
                if (hasEntities || hasPairs || hasSources) {
                    sb.append("**Semantische Interpretation**\n\n");
                    if (hasEntities) {
                        sb.append("Entities: ").append(String.join(", ", first.entityNames())).append("\n\n");
                    }
                    if (hasPairs) {
                        sb.append("Relationships: ").append(String.join(" | ", first.relationshipPairs())).append("\n\n");
                    }
                    if (hasSources) {
                        sb.append("| Quelle | Dokumente |\n|---|---|\n");
                        first.documentsBySource().forEach((src, docs) -> {
                            if (!docs.isEmpty()) {
                                sb.append("| ").append(src).append(" | ").append(String.join(", ", docs)).append(" |\n");
                            }
                        });
                        sb.append("\n");
                    }
                }
            }

            sb.append("#### LLM — Läufe\n\n");
            sb.append("| Lauf | Recall | Precision | F1 | Hit | Dauer (s) | Gefunden |\n|---|---|---|---|---|---|---|\n");
            for (int i = 0; i < r.llmRuns().size(); i++) {
                LlmRunResult l = r.llmRuns().get(i);
                sb.append(String.format("| %d | %.2f | %.2f | %.2f | %s | %.2f | %s |%n",
                        i + 1, l.recall(), l.precision(), l.f1(),
                        l.hit() ? "✓" : "✗",
                        l.durationMs() / 1000.0,
                        joinList(l.retrievedDocuments())));
            }
            sb.append("\n");

            // LLM-Antworten — alle Läufe als <details>-Blöcke
            for (int i = 0; i < r.llmRuns().size(); i++) {
                LlmRunResult l = r.llmRuns().get(i);
                if (l.responseText() != null && !l.responseText().isBlank()) {
                    sb.append(String.format(
                            "<details><summary>Lauf %d — Recall: %.2f  F1: %.2f  (%.1fs)</summary>%n%n",
                            i + 1, l.recall(), l.f1(), l.durationMs() / 1000.0));
                    sb.append("```\n").append(l.responseText()).append("\n```\n\n</details>\n\n");
                }
            }
        }

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private void appendModeComparison(StringBuilder sb, List<AggregatedEvalResult> results) {
        Map<String, List<AggregatedEvalResult>> byMode = new LinkedHashMap<>();
        for (AggregatedEvalResult r : results) {
            byMode.computeIfAbsent(r.queryMode(), k -> new ArrayList<>()).add(r);
        }
        if (byMode.size() <= 1) return;

        double bestLlmF1      = byMode.values().stream().flatMap(List::stream).mapToDouble(r -> r.llmMetrics().avgF1()).max().orElse(0);
        double bestGraphRecall = byMode.values().stream().flatMap(List::stream).mapToDouble(r -> r.graphMetrics().avgRecallAtK()).max().orElse(0);

        sb.append("### Graph-Retrieval\n\n");
        sb.append("| Mode | Ø MRR | ±σ | Ø NDCG@k | ±σ | Ø Recall@k | ±σ |\n|---|---|---|---|---|---|---|\n");
        byMode.forEach((mode, rs) -> {
            double[] mrrArr    = rs.stream().mapToDouble(r -> r.graphMetrics().avgMrr()).toArray();
            double[] ndcgArr   = rs.stream().mapToDouble(r -> r.graphMetrics().avgNdcgAtK()).toArray();
            double[] recallArr = rs.stream().mapToDouble(r -> r.graphMetrics().avgRecallAtK()).toArray();
            double recall = avg(recallArr);
            boolean best  = Math.abs(recall - bestGraphRecall) < 0.001;
            sb.append(String.format("| %s%s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    mode, best ? " ★" : "",
                    avg(mrrArr), ReportData.stdDev(mrrArr),
                    avg(ndcgArr), ReportData.stdDev(ndcgArr),
                    recall, ReportData.stdDev(recallArr)));
        });

        sb.append("\n### LLM\n\n");
        sb.append("| Mode | Ø Recall | ±σ | Ø Precision | ±σ | Ø F1 | ±σ | Ø Hit-Rate | ±σ | Ø MRR | ±σ |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        byMode.forEach((mode, rs) -> {
            double[] recallArr = rs.stream().mapToDouble(r -> r.llmMetrics().avgRecall()).toArray();
            double[] precArr   = rs.stream().mapToDouble(r -> r.llmMetrics().avgPrecision()).toArray();
            double[] f1Arr     = rs.stream().mapToDouble(r -> r.llmMetrics().avgF1()).toArray();
            double[] hitArr    = rs.stream().mapToDouble(r -> r.llmMetrics().hitRate()).toArray();
            double[] mrrArr    = rs.stream().mapToDouble(r -> r.llmMetrics().avgMrr()).toArray();
            double f1    = avg(f1Arr);
            boolean best = Math.abs(f1 - bestLlmF1) < 0.001;
            sb.append(String.format("| **%s**%s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    mode, best ? " ★" : "",
                    avg(recallArr), ReportData.stdDev(recallArr),
                    avg(precArr),   ReportData.stdDev(precArr),
                    f1,             ReportData.stdDev(f1Arr),
                    avg(hitArr),    ReportData.stdDev(hitArr),
                    avg(mrrArr),    ReportData.stdDev(mrrArr)));
        });
    }

    private static double avg(double[] values) {
        return java.util.Arrays.stream(values).average().orElse(0.0);
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
