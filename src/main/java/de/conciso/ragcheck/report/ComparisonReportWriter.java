package de.conciso.ragcheck.report;

import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Erzeugt einen Vergleichsreport über mehrere ragcheck-Testläufe.
 * Eingabe: Liste von ComparisonEntry (je ein Eintrag pro Lauf).
 * Ausgabe: comparison.md + comparison.html im Reports-Verzeichnis.
 */
@Component
public class ComparisonReportWriter {

    public record ComparisonEntry(
            String timestamp,
            String runLabel,
            Map<String, String> runParameters,
            String queryMode,
            int topK,
            int runsPerTestCase,
            double avgGraphMrr,
            double avgGraphNdcgAtK,
            double avgGraphRecallAtK,
            double avgLlmRecall,
            double avgLlmPrecision,
            double avgLlmF1,
            double avgLlmHitRate,
            double avgLlmMrr
    ) {}

    public void write(List<ComparisonEntry> entries, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);

        List<ComparisonEntry> sorted = entries.stream()
                .sorted(Comparator.comparingDouble(ComparisonEntry::avgLlmF1).reversed())
                .toList();

        Path mdPath = outputDir.resolve("comparison.md");
        Files.writeString(mdPath, buildMarkdown(sorted), StandardCharsets.UTF_8);

        Path htmlPath = outputDir.resolve("comparison.html");
        Files.writeString(htmlPath, buildHtml(sorted), StandardCharsets.UTF_8);
    }

    // -------------------------------------------------------------------------

    private String buildMarkdown(List<ComparisonEntry> entries) {
        StringBuilder sb = new StringBuilder();
        sb.append("# RAGChecker — Parametervergleich\n\n");
        sb.append("Sortiert nach Ø LLM-F1 (absteigend). Bester Wert je Spalte mit **★** markiert.\n\n");

        // Beste Werte ermitteln
        double bestGraphMrr      = best(entries, ComparisonEntry::avgGraphMrr);
        double bestGraphNdcg     = best(entries, ComparisonEntry::avgGraphNdcgAtK);
        double bestGraphRecall   = best(entries, ComparisonEntry::avgGraphRecallAtK);
        double bestLlmRecall     = best(entries, ComparisonEntry::avgLlmRecall);
        double bestLlmPrecision  = best(entries, ComparisonEntry::avgLlmPrecision);
        double bestLlmF1         = best(entries, ComparisonEntry::avgLlmF1);
        double bestLlmHitRate    = best(entries, ComparisonEntry::avgLlmHitRate);
        double bestLlmMrr        = best(entries, ComparisonEntry::avgLlmMrr);

        sb.append("## Graph-Retrieval\n\n");
        sb.append("| # | Timestamp | Label | Parameter | Mode | Top-K | Ø MRR | Ø NDCG@k | Ø Recall@k |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|\n");
        for (int i = 0; i < entries.size(); i++) {
            ComparisonEntry e = entries.get(i);
            sb.append(String.format("| %d | %s | %s | %s | %s | %d | %s | %s | %s |\n",
                    i + 1,
                    e.timestamp(),
                    e.runLabel().isBlank() ? "—" : e.runLabel(),
                    formatParams(e.runParameters()),
                    e.queryMode(),
                    e.topK(),
                    mark(e.avgGraphMrr(), bestGraphMrr),
                    mark(e.avgGraphNdcgAtK(), bestGraphNdcg),
                    mark(e.avgGraphRecallAtK(), bestGraphRecall)));
        }

        sb.append("\n## LLM\n\n");
        sb.append("| # | Timestamp | Label | Parameter | Mode | Top-K | Ø Recall | Ø Precision | Ø F1 | Ø Hit-Rate | Ø MRR |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (int i = 0; i < entries.size(); i++) {
            ComparisonEntry e = entries.get(i);
            sb.append(String.format("| %d | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s |\n",
                    i + 1,
                    e.timestamp(),
                    e.runLabel().isBlank() ? "—" : e.runLabel(),
                    formatParams(e.runParameters()),
                    e.queryMode(),
                    e.topK(),
                    mark(e.avgLlmRecall(), bestLlmRecall),
                    mark(e.avgLlmPrecision(), bestLlmPrecision),
                    mark(e.avgLlmF1(), bestLlmF1),
                    mark(e.avgLlmHitRate(), bestLlmHitRate),
                    mark(e.avgLlmMrr(), bestLlmMrr)));
        }

        if (!entries.isEmpty()) {
            sb.append("\n## Bester Run\n\n");
            ComparisonEntry top = entries.get(0);
            sb.append("**Label:** ").append(top.runLabel().isBlank() ? "—" : top.runLabel()).append("  \n");
            sb.append("**Timestamp:** ").append(top.timestamp()).append("  \n");
            sb.append("**Query Mode:** ").append(top.queryMode())
              .append(" | **Top-K:** ").append(top.topK()).append("  \n");
            if (!top.runParameters().isEmpty()) {
                sb.append("**Parameter:**  \n");
                top.runParameters().forEach((k, v) ->
                        sb.append("- `").append(k).append("` = `").append(v).append("`\n"));
            }
            sb.append(String.format("%nGraph: MRR=%.2f  NDCG=%.2f  Recall=%.2f  \n",
                    top.avgGraphMrr(), top.avgGraphNdcgAtK(), top.avgGraphRecallAtK()));
            sb.append(String.format("LLM:   Recall=%.2f  Precision=%.2f  F1=%.2f  HitRate=%.2f  MRR=%.2f%n",
                    top.avgLlmRecall(), top.avgLlmPrecision(), top.avgLlmF1(),
                    top.avgLlmHitRate(), top.avgLlmMrr()));
        }

        return sb.toString();
    }

    private String buildHtml(List<ComparisonEntry> entries) {
        // Beste Werte für Spalten-Highlighting
        double bestGraphMrr      = best(entries, ComparisonEntry::avgGraphMrr);
        double bestGraphNdcg     = best(entries, ComparisonEntry::avgGraphNdcgAtK);
        double bestGraphRecall   = best(entries, ComparisonEntry::avgGraphRecallAtK);
        double bestLlmRecall     = best(entries, ComparisonEntry::avgLlmRecall);
        double bestLlmPrecision  = best(entries, ComparisonEntry::avgLlmPrecision);
        double bestLlmF1         = best(entries, ComparisonEntry::avgLlmF1);
        double bestLlmHitRate    = best(entries, ComparisonEntry::avgLlmHitRate);
        double bestLlmMrr        = best(entries, ComparisonEntry::avgLlmMrr);

        StringBuilder sb = new StringBuilder();
        sb.append("""
                <!DOCTYPE html>
                <html lang="de">
                <head>
                <meta charset="UTF-8">
                <title>RAGChecker — Parametervergleich</title>
                <style>
                  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                         margin: 2rem; background: #f8f9fa; color: #212529; }
                  h1 { color: #1a1a2e; }
                  h2 { color: #16213e; margin-top: 2rem; }
                  .subtitle { color: #6c757d; margin-bottom: 2rem; }
                  table { border-collapse: collapse; width: 100%; background: #fff;
                          box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 6px;
                          overflow: hidden; margin-bottom: 2rem; }
                  th { background: #1a1a2e; color: #fff; padding: .6rem .8rem;
                       text-align: left; font-size: .85rem; cursor: pointer; user-select: none; }
                  th:hover { background: #2d2d5e; }
                  th::after { content: ' ↕'; opacity: .4; }
                  th.asc::after { content: ' ↑'; opacity: 1; }
                  th.desc::after { content: ' ↓'; opacity: 1; }
                  td { padding: .5rem .8rem; font-size: .85rem; border-bottom: 1px solid #dee2e6; }
                  tr:last-child td { border-bottom: none; }
                  tr:hover td { background: #f1f3f5; }
                  tr.best td { background: #e8f5e9; }
                  .val-good  { color: #2e7d32; font-weight: 600; }
                  .val-mid   { color: #f57c00; }
                  .val-bad   { color: #c62828; }
                  .best-mark { font-size: .7rem; color: #1b5e20; margin-left: .2rem; }
                  .params { font-size: .75rem; color: #495057; }
                  .summary-box { background: #fff; border-radius: 6px;
                                 box-shadow: 0 1px 4px rgba(0,0,0,.1);
                                 padding: 1.2rem 1.5rem; margin-bottom: 2rem; max-width: 600px; }
                  .summary-box h3 { margin: 0 0 .5rem; color: #1a1a2e; }
                  .summary-box p  { margin: .2rem 0; }
                </style>
                </head>
                <body>
                <h1>RAGChecker — Parametervergleich</h1>
                <p class="subtitle">Sortiert nach Ø LLM-F1 (absteigend). ★ = bester Wert in dieser Spalte.</p>
                """);

        if (!entries.isEmpty()) {
            ComparisonEntry top = entries.get(0);
            sb.append("<div class=\"summary-box\">\n<h3>Bester Run</h3>\n");
            sb.append("<p><strong>Label:</strong> ").append(esc(top.runLabel().isBlank() ? "—" : top.runLabel())).append("</p>\n");
            sb.append("<p><strong>Timestamp:</strong> ").append(esc(top.timestamp())).append("</p>\n");
            sb.append("<p><strong>Query Mode:</strong> ").append(esc(top.queryMode()))
              .append(" &nbsp; <strong>Top-K:</strong> ").append(top.topK()).append("</p>\n");
            if (!top.runParameters().isEmpty()) {
                sb.append("<p><strong>Parameter:</strong> ").append(esc(formatParams(top.runParameters()))).append("</p>\n");
            }
            sb.append(String.format("<p>Graph: MRR=%.2f &nbsp; NDCG=%.2f &nbsp; Recall=%.2f</p>%n",
                    top.avgGraphMrr(), top.avgGraphNdcgAtK(), top.avgGraphRecallAtK()));
            sb.append(String.format("<p>LLM: Recall=%.2f &nbsp; Precision=%.2f &nbsp; <strong>F1=%.2f</strong> &nbsp; HitRate=%.2f &nbsp; MRR=%.2f</p>%n",
                    top.avgLlmRecall(), top.avgLlmPrecision(), top.avgLlmF1(),
                    top.avgLlmHitRate(), top.avgLlmMrr()));
            sb.append("</div>\n");
        }

        sb.append("<h2>Graph-Retrieval</h2>\n");
        sb.append("<table id=\"graphTable\">\n<thead><tr>\n");
        for (String h : List.of("#", "Timestamp", "Label", "Parameter", "Mode", "Top-K",
                "Ø MRR", "Ø NDCG@k", "Ø Recall@k")) {
            sb.append("<th>").append(h).append("</th>\n");
        }
        sb.append("</tr></thead>\n<tbody>\n");
        for (int i = 0; i < entries.size(); i++) {
            ComparisonEntry e = entries.get(i);
            sb.append(i == 0 ? "<tr class=\"best\">" : "<tr>");
            sb.append("<td>").append(i + 1).append("</td>");
            sb.append("<td>").append(esc(e.timestamp())).append("</td>");
            sb.append("<td>").append(esc(e.runLabel().isBlank() ? "—" : e.runLabel())).append("</td>");
            sb.append("<td class=\"params\">").append(esc(formatParams(e.runParameters()))).append("</td>");
            sb.append("<td>").append(esc(e.queryMode())).append("</td>");
            sb.append("<td>").append(e.topK()).append("</td>");
            sb.append(valCell(e.avgGraphMrr(), bestGraphMrr));
            sb.append(valCell(e.avgGraphNdcgAtK(), bestGraphNdcg));
            sb.append(valCell(e.avgGraphRecallAtK(), bestGraphRecall));
            sb.append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        sb.append("<h2>LLM</h2>\n");
        sb.append("<table id=\"llmTable\">\n<thead><tr>\n");
        for (String h : List.of("#", "Timestamp", "Label", "Parameter", "Mode", "Top-K",
                "Ø Recall", "Ø Precision", "Ø F1", "Ø Hit-Rate", "Ø MRR")) {
            sb.append("<th>").append(h).append("</th>\n");
        }
        sb.append("</tr></thead>\n<tbody>\n");
        for (int i = 0; i < entries.size(); i++) {
            ComparisonEntry e = entries.get(i);
            sb.append(i == 0 ? "<tr class=\"best\">" : "<tr>");
            sb.append("<td>").append(i + 1).append("</td>");
            sb.append("<td>").append(esc(e.timestamp())).append("</td>");
            sb.append("<td>").append(esc(e.runLabel().isBlank() ? "—" : e.runLabel())).append("</td>");
            sb.append("<td class=\"params\">").append(esc(formatParams(e.runParameters()))).append("</td>");
            sb.append("<td>").append(esc(e.queryMode())).append("</td>");
            sb.append("<td>").append(e.topK()).append("</td>");
            sb.append(valCell(e.avgLlmRecall(), bestLlmRecall));
            sb.append(valCell(e.avgLlmPrecision(), bestLlmPrecision));
            sb.append(valCell(e.avgLlmF1(), bestLlmF1));
            sb.append(valCell(e.avgLlmHitRate(), bestLlmHitRate));
            sb.append(valCell(e.avgLlmMrr(), bestLlmMrr));
            sb.append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        sb.append("""
                <script>
                document.querySelectorAll('table').forEach(table => {
                  table.querySelectorAll('th').forEach((th, colIdx) => {
                    th.addEventListener('click', () => {
                      const tbody = table.querySelector('tbody');
                      const rows = Array.from(tbody.querySelectorAll('tr'));
                      const asc = th.classList.toggle('asc');
                      th.classList.toggle('desc', !asc);
                      table.querySelectorAll('th').forEach(other => {
                        if (other !== th) { other.classList.remove('asc', 'desc'); }
                      });
                      rows.sort((a, b) => {
                        const av = a.cells[colIdx].textContent.trim();
                        const bv = b.cells[colIdx].textContent.trim();
                        const an = parseFloat(av), bn = parseFloat(bv);
                        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
                        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
                      });
                      rows.forEach(r => tbody.appendChild(r));
                    });
                  });
                });
                </script>
                </body></html>
                """);

        return sb.toString();
    }

    // -------------------------------------------------------------------------

    private String valCell(double value, double best) {
        String cls = value >= 0.8 ? "val-good" : value >= 0.5 ? "val-mid" : "val-bad";
        String star = Math.abs(value - best) < 0.001 ? "<span class=\"best-mark\">★</span>" : "";
        return String.format("<td class=\"%s\">%.2f%s</td>", cls, value, star);
    }

    private String mark(double value, double best) {
        String formatted = String.format("%.2f", value);
        return Math.abs(value - best) < 0.001 ? "**" + formatted + " ★**" : formatted;
    }

    private String formatParams(Map<String, String> params) {
        if (params == null || params.isEmpty()) return "—";
        StringBuilder sb = new StringBuilder();
        params.forEach((k, v) -> {
            if (!sb.isEmpty()) sb.append(", ");
            sb.append(k).append("=").append(v);
        });
        return sb.toString();
    }

    private String esc(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    @FunctionalInterface
    private interface MetricExtractor {
        double extract(ComparisonEntry e);
    }

    private double best(List<ComparisonEntry> entries, MetricExtractor extractor) {
        OptionalDouble max = entries.stream().mapToDouble(extractor::extract).max();
        return max.orElse(0.0);
    }
}
