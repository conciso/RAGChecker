package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.AggregatedEvalResult;
import de.conciso.ragcheck.model.EvalResult;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;

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
        sb.append("| Parameter | Wert |\n");
        sb.append("|---|---|\n");
        sb.append("| Query Mode | ").append(data.queryMode()).append(" |\n");
        sb.append("| Top-K | ").append(data.topK()).append(" |\n");
        sb.append("| Läufe je Testfall | ").append(data.runsPerTestCase()).append(" |\n");
        sb.append("| Testfälle | ").append(data.testCasesPath()).append(" |\n\n");

        // Zusammenfassung
        sb.append("## Zusammenfassung\n\n");
        sb.append("| Metrik | Wert |\n");
        sb.append("|---|---|\n");
        sb.append(String.format("| Ø Recall | %.2f |%n", data.avgRecall()));
        sb.append(String.format("| Ø Precision | %.2f |%n", data.avgPrecision()));
        sb.append(String.format("| Ø F1 | %.2f |%n", data.avgF1()));
        sb.append(String.format("| Ø Hit-Rate | %.2f |%n", data.avgHitRate()));
        sb.append(String.format("| Ø MRR | %.2f |%n", data.avgMrr()));
        sb.append("| Testfälle gesamt | ").append(data.results().size()).append(" |\n\n");

        // Ergebnisse-Tabelle
        sb.append("## Ergebnisse\n\n");
        sb.append("| ID | Status | Recall | ±σ | Precision | ±σ | F1 | ±σ | Hit-Rate | MRR |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|---|\n");
        for (AggregatedEvalResult r : data.results()) {
            String status = statusEmoji(r.avgF1());
            sb.append(String.format("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |%n",
                    r.testCaseId(), status,
                    r.avgRecall(), r.stdDevRecall(),
                    r.avgPrecision(), r.stdDevPrecision(),
                    r.avgF1(), r.stdDevF1(),
                    r.hitRate(), r.mrr()));
        }
        sb.append("\n");

        // Details
        sb.append("## Details\n\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append("### ").append(r.testCaseId()).append("\n\n");
            sb.append("**Prompt:** ").append(r.prompt()).append("\n\n");
            sb.append("**Erwartet:** ").append(joinList(r.expectedDocuments())).append("\n\n");

            sb.append("| Lauf | Recall | Precision | F1 | Gefunden |\n");
            sb.append("|---|---|---|---|---|\n");
            for (int i = 0; i < r.runResults().size(); i++) {
                EvalResult run = r.runResults().get(i);
                sb.append(String.format("| %d | %.2f | %.2f | %.2f | %s |%n",
                        i + 1, run.recall(), run.precision(), run.f1(),
                        joinList(run.retrievedDocuments())));
            }
            sb.append("\n");

            // Keywords aus dem letzten Lauf (repräsentativ)
            EvalResult lastRun = r.runResults().get(r.runResults().size() - 1);
            if (lastRun.queryResult() != null) {
                List<String> high = lastRun.queryResult().highLevelKeywords();
                List<String> low = lastRun.queryResult().lowLevelKeywords();
                if (high != null && !high.isEmpty()) {
                    sb.append("**Keywords (high):** ").append(String.join(", ", high)).append("\n\n");
                }
                if (low != null && !low.isEmpty()) {
                    sb.append("**Keywords (low):** ").append(String.join(", ", low)).append("\n\n");
                }
            }
        }

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private String statusEmoji(double f1) {
        if (f1 >= 0.8) return "✅";
        if (f1 >= 0.5) return "⚠️";
        return "❌";
    }

    private String joinList(List<String> list) {
        if (list == null || list.isEmpty()) return "—";
        return String.join(", ", list);
    }
}
