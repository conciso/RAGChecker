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
import java.util.stream.Collectors;

@Component
public class HtmlReportWriter {

    private static final DateTimeFormatter DISPLAY_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    public void write(ReportData data, Path path) throws IOException {
        Files.writeString(path, buildHtml(data), StandardCharsets.UTF_8);
    }

    private String buildHtml(ReportData data) {
        String ts = DISPLAY_FORMAT.format(data.timestamp());
        StringBuilder sb = new StringBuilder();

        sb.append("<!DOCTYPE html>\n<html lang=\"de\">\n<head>\n");
        sb.append("<meta charset=\"UTF-8\">\n");
        sb.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        sb.append("<title>RAGChecker Report — ").append(escape(ts)).append("</title>\n");
        sb.append("<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4\"></script>\n");
        sb.append("<script src=\"https://cdn.jsdelivr.net/npm/@sgratzl/chartjs-chart-boxplot@4\"></script>\n");
        sb.append(css());
        sb.append("</head>\n<body>\n");

        // Header
        sb.append("<header>\n<h1>RAGChecker Report</h1>\n");
        sb.append("<p class=\"ts\">").append(escape(ts)).append("</p>\n");
        sb.append("<div class=\"cfg\">\n");
        if (data.runLabel() != null && !data.runLabel().isBlank()) {
            sb.append(cfgItem("Run-Label", data.runLabel()));
        }
        sb.append(cfgItem("Query Modes", String.join(", ", data.queryModes())));
        sb.append(cfgItem("Top-K", String.valueOf(data.topK())));
        sb.append(cfgItem("Läufe", String.valueOf(data.runsPerTestCase())));
        sb.append(cfgItem("Testfälle", data.testCasesPath()));
        if (data.runParameters() != null && !data.runParameters().isEmpty()) {
            data.runParameters().forEach((k, v) -> sb.append(cfgItem(k, v)));
        }
        sb.append("</div>\n</header>\n");

        // Summary cards — two groups
        sb.append("<section>\n<h2>Graph-Retrieval (/query/data)</h2>\n<div class=\"cards\">\n");
        sb.append(card("Ø MRR", data.avgGraphMrr()));
        sb.append(card("Ø NDCG@k", data.avgGraphNdcgAtK()));
        sb.append(card("Ø Recall@k", data.avgGraphRecallAtK()));
        sb.append("</div>\n</section>\n");

        sb.append("<section>\n<h2>LLM (/query)</h2>\n<div class=\"cards\">\n");
        sb.append(card("Ø Recall", data.avgLlmRecall()));
        sb.append(card("Ø Precision", data.avgLlmPrecision()));
        sb.append(card("Ø F1", data.avgLlmF1()));
        sb.append(card("Ø Hit-Rate", data.avgLlmHitRate()));
        sb.append(card("Ø MRR", data.avgLlmMrr()));
        sb.append("</div>\n</section>\n");

        // Mode comparison (only if more than one mode)
        appendModeComparisonSection(sb, data.results());

        // Bar chart
        sb.append("<section>\n<h2>Metriken je Testfall</h2>\n");
        sb.append("<div class=\"chart-wrap\"><canvas id=\"chart\"></canvas></div>\n</section>\n");

        // Boxplots (nur bei mehr als 1 Lauf sinnvoll)
        if (data.runsPerTestCase() > 1) {
            sb.append("<section>\n<h2>Verteilung über Läufe (Boxplots)</h2>\n");
            sb.append("<p class=\"sub\">Jede Box zeigt die Verteilung der Metrikwerte über alle Läufe je Testfall.</p>\n");
            sb.append("<div class=\"chart-grid\">\n");
            sb.append("<div class=\"chart-wrap\"><canvas id=\"bpGraph\"></canvas></div>\n");
            sb.append("<div class=\"chart-wrap\"><canvas id=\"bpLlm\"></canvas></div>\n");
            sb.append("</div>\n</section>\n");
        }

        // Results tables
        sb.append("<section>\n<h2>Ergebnisse — Graph-Retrieval</h2>\n");
        sb.append("<table><thead><tr>");
        sb.append("<th>Mode</th><th>ID</th><th>Status</th><th>MRR</th><th>±σ</th><th>NDCG@k</th><th>±σ</th><th>Recall@k</th><th>±σ</th>");
        sb.append("</tr></thead><tbody>\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append(String.format(
                    "<tr class=\"%s\"><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td class=\"m\">%.2f</td><td>%.2f</td><td class=\"m\">%.2f</td><td>%.2f</td><td class=\"m\">%.2f</td></tr>%n",
                    rc(r.graphMetrics().avgMrr()), escape(r.queryMode()), escape(r.testCaseId()), emoji(r.graphMetrics().avgMrr()),
                    r.graphMetrics().avgMrr(), r.graphMetrics().stdDevMrr(),
                    r.graphMetrics().avgNdcgAtK(), r.graphMetrics().stdDevNdcgAtK(),
                    r.graphMetrics().avgRecallAtK(), r.graphMetrics().stdDevRecallAtK()));
        }
        sb.append("</tbody></table>\n</section>\n");

        sb.append("<section>\n<h2>Ergebnisse — LLM</h2>\n");
        sb.append("<table><thead><tr>");
        sb.append("<th>Mode</th><th>ID</th><th>Status</th><th>Recall</th><th>±σ</th><th>Precision</th><th>±σ</th><th>F1</th><th>±σ</th><th>Hit-Rate</th><th>MRR</th>");
        sb.append("</tr></thead><tbody>\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append(String.format(
                    "<tr class=\"%s\"><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td class=\"m\">%.2f</td><td>%.2f</td><td class=\"m\">%.2f</td><td>%.2f</td><td class=\"m\">%.2f</td><td>%.2f</td><td>%.2f</td></tr>%n",
                    rc(r.llmMetrics().avgF1()), escape(r.queryMode()), escape(r.testCaseId()), emoji(r.llmMetrics().avgF1()),
                    r.llmMetrics().avgRecall(), r.llmMetrics().stdDevRecall(),
                    r.llmMetrics().avgPrecision(), r.llmMetrics().stdDevPrecision(),
                    r.llmMetrics().avgF1(), r.llmMetrics().stdDevF1(),
                    r.llmMetrics().hitRate(), r.llmMetrics().avgMrr()));
        }
        sb.append("</tbody></table>\n</section>\n");

        // Details
        sb.append("<section>\n<h2>Details</h2>\n");
        for (AggregatedEvalResult r : data.results()) {
            sb.append("<details>\n<summary><strong>").append(escape(r.testCaseId()))
                    .append(" / ").append(escape(r.queryMode()))
                    .append("</strong> — Graph MRR: ").append(String.format("%.2f", r.graphMetrics().avgMrr()))
                    .append(" &nbsp; LLM F1: ").append(String.format("%.2f", r.llmMetrics().avgF1()))
                    .append("</summary>\n<div class=\"db\">\n");

            sb.append("<p><strong>Prompt:</strong> ").append(escape(r.prompt())).append("</p>\n");
            sb.append("<p><strong>Erwartet:</strong> ").append(escape(String.join(", ", r.expectedDocuments()))).append("</p>\n");

            // Rank table per expected doc
            sb.append("<h4>Ränge je erwartetem Dokument</h4>\n");
            sb.append("<table class=\"dt\"><thead><tr><th>Dokument</th>");
            for (int i = 1; i <= r.runs(); i++) sb.append("<th>Lauf ").append(i).append("</th>");
            sb.append("</tr></thead><tbody>\n");
            for (Map.Entry<String, List<Integer>> e : r.graphMetrics().ranksByDoc().entrySet()) {
                sb.append("<tr><td>").append(escape(e.getKey())).append("</td>");
                for (int rank : e.getValue()) {
                    String cls = rank == 0 ? "not-found" : rank <= 3 ? "top" : "";
                    sb.append("<td class=\"").append(cls).append("\">")
                            .append(rank == 0 ? "—" : rank).append("</td>");
                }
                sb.append("</tr>\n");
            }
            sb.append("</tbody></table>\n");

            // Graph runs
            sb.append("<h4>Graph-Retrieval — Läufe</h4>\n");
            sb.append("<table class=\"dt\"><thead><tr><th>Lauf</th><th>MRR</th><th>NDCG@k</th><th>Recall@k</th><th>Dauer (s)</th><th>Gefunden (Top 5)</th></tr></thead><tbody>\n");
            for (int i = 0; i < r.graphRuns().size(); i++) {
                GraphRetrievalRunResult g = r.graphRuns().get(i);
                String top5 = g.retrievedDocuments().stream().limit(5)
                        .map(this::escape).collect(Collectors.joining(", "));
                sb.append(String.format("<tr><td>%d</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%s</td></tr>%n",
                        i + 1, g.mrr(), g.ndcgAtK(), g.recallAtK(), g.durationMs() / 1000.0, top5.isEmpty() ? "—" : top5));
            }
            sb.append("</tbody></table>\n");

            // LLM runs
            sb.append("<h4>LLM — Läufe</h4>\n");
            sb.append("<table class=\"dt\"><thead><tr><th>Lauf</th><th>Recall</th><th>Precision</th><th>F1</th><th>Hit</th><th>Dauer (s)</th><th>Gefunden</th></tr></thead><tbody>\n");
            for (int i = 0; i < r.llmRuns().size(); i++) {
                LlmRunResult l = r.llmRuns().get(i);
                sb.append(String.format("<tr><td>%d</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%s</td><td>%.2f</td><td>%s</td></tr>%n",
                        i + 1, l.recall(), l.precision(), l.f1(),
                        l.hit() ? "✓" : "✗", l.durationMs() / 1000.0, escape(joinList(l.retrievedDocuments()))));
            }
            sb.append("</tbody></table>\n");

            // LLM response text from last run
            if (!r.llmRuns().isEmpty()) {
                String resp = r.llmRuns().get(r.llmRuns().size() - 1).responseText();
                if (resp != null && !resp.isBlank()) {
                    sb.append("<h4>LLM-Antwort (letzter Lauf)</h4>\n");
                    sb.append("<pre class=\"resp\">").append(escape(resp)).append("</pre>\n");
                }
            }

            sb.append("</div>\n</details>\n");
        }
        sb.append("</section>\n");

        sb.append(chartScript(data));
        if (data.runsPerTestCase() > 1) {
            sb.append(boxplotScript(data));
        }
        sb.append("</body>\n</html>\n");
        return sb.toString();
    }

    private void appendModeComparisonSection(StringBuilder sb, List<AggregatedEvalResult> results) {
        Map<String, List<AggregatedEvalResult>> byMode = new LinkedHashMap<>();
        for (AggregatedEvalResult r : results) {
            byMode.computeIfAbsent(r.queryMode(), k -> new ArrayList<>()).add(r);
        }
        if (byMode.size() <= 1) return;

        double bestLlmF1       = byMode.values().stream().flatMap(List::stream).mapToDouble(r -> r.llmMetrics().avgF1()).max().orElse(0);
        double bestGraphRecall = byMode.values().stream().flatMap(List::stream).mapToDouble(r -> r.graphMetrics().avgRecallAtK()).max().orElse(0);

        sb.append("<section>\n<h2>Modus-Vergleich</h2>\n");
        sb.append("<p class=\"sub\">Ø über alle Testfälle. ★ = bester Modus in der jeweiligen Hauptmetrik.</p>\n");

        // Graph table
        sb.append("<h3>Graph-Retrieval</h3>\n");
        sb.append("<table><thead><tr><th>Mode</th><th>Ø MRR</th><th>Ø NDCG@k</th><th>Ø Recall@k</th></tr></thead><tbody>\n");
        byMode.forEach((mode, rs) -> {
            double mrr    = rs.stream().mapToDouble(r -> r.graphMetrics().avgMrr()).average().orElse(0);
            double ndcg   = rs.stream().mapToDouble(r -> r.graphMetrics().avgNdcgAtK()).average().orElse(0);
            double recall = rs.stream().mapToDouble(r -> r.graphMetrics().avgRecallAtK()).average().orElse(0);
            boolean best  = Math.abs(recall - bestGraphRecall) < 0.001;
            String rowCls = best ? " class=\"best-mode\"" : "";
            sb.append(String.format("<tr%s><td><strong>%s</strong>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>%n",
                    rowCls, escape(mode), best ? " ★" : "", mrr, ndcg, recall));
        });
        sb.append("</tbody></table>\n");

        // LLM table
        sb.append("<h3>LLM</h3>\n");
        sb.append("<table><thead><tr><th>Mode</th><th>Ø Recall</th><th>Ø Precision</th><th>Ø F1</th><th>Ø Hit-Rate</th><th>Ø MRR</th></tr></thead><tbody>\n");
        byMode.forEach((mode, rs) -> {
            double recall = rs.stream().mapToDouble(r -> r.llmMetrics().avgRecall()).average().orElse(0);
            double prec   = rs.stream().mapToDouble(r -> r.llmMetrics().avgPrecision()).average().orElse(0);
            double f1     = rs.stream().mapToDouble(r -> r.llmMetrics().avgF1()).average().orElse(0);
            double hit    = rs.stream().mapToDouble(r -> r.llmMetrics().hitRate()).average().orElse(0);
            double mrr    = rs.stream().mapToDouble(r -> r.llmMetrics().avgMrr()).average().orElse(0);
            boolean best  = Math.abs(f1 - bestLlmF1) < 0.001;
            String rowCls = best ? " class=\"best-mode\"" : "";
            sb.append(String.format("<tr%s><td><strong>%s</strong>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>%n",
                    rowCls, escape(mode), best ? " ★" : "", recall, prec, f1, hit, mrr));
        });
        sb.append("</tbody></table>\n");
        sb.append("</section>\n");
    }

    private String chartScript(ReportData data) {
        String labels     = join(data.results(), r -> "\"" + (r.testCaseId() + " / " + r.queryMode()).replace("\"", "\\\"") + "\"");
        String graphMrr   = join(data.results(), r -> fmt(r.graphMetrics().avgMrr()));
        String graphNdcg  = join(data.results(), r -> fmt(r.graphMetrics().avgNdcgAtK()));
        String graphRec   = join(data.results(), r -> fmt(r.graphMetrics().avgRecallAtK()));
        String llmRecall  = join(data.results(), r -> fmt(r.llmMetrics().avgRecall()));
        String llmPrec    = join(data.results(), r -> fmt(r.llmMetrics().avgPrecision()));
        String llmF1      = join(data.results(), r -> fmt(r.llmMetrics().avgF1()));

        return "<script>\nnew Chart(document.getElementById('chart').getContext('2d'), {\n"
                + "  type: 'bar',\n"
                + "  data: {\n"
                + "    labels: [" + labels + "],\n"
                + "    datasets: [\n"
                + ds("Graph MRR",      graphMrr,  "rgba(54,162,235,0.7)")
                + ds("Graph NDCG@k",   graphNdcg, "rgba(54,162,235,0.4)")
                + ds("Graph Recall@k", graphRec,  "rgba(54,162,235,0.2)")
                + ds("LLM Recall",     llmRecall, "rgba(255,159,64,0.7)")
                + ds("LLM Precision",  llmPrec,   "rgba(75,192,192,0.7)")
                + ds("LLM F1",         llmF1,     "rgba(153,102,255,0.7)")
                + "    ]\n  },\n"
                + "  options: { responsive: true, scales: { y: { beginAtZero: true, max: 1.0 } } }\n"
                + "});\n</script>\n";
    }

    private String boxplotScript(ReportData data) {
        // Labels: Testfall-ID / Mode
        String labels = join(data.results(), r -> "\"" + (r.testCaseId() + " / " + r.queryMode()).replace("\"", "\\\"") + "\"");

        // Für jeden Testfall: Array der Lauf-Werte → [[run1,run2,...], [run1,run2,...], ...]
        String graphMrrData    = bpData(data.results(), r -> r.graphRuns().stream()
                .map(g -> fmt(g.mrr())).collect(Collectors.joining(",")));
        String graphNdcgData   = bpData(data.results(), r -> r.graphRuns().stream()
                .map(g -> fmt(g.ndcgAtK())).collect(Collectors.joining(",")));
        String graphRecallData = bpData(data.results(), r -> r.graphRuns().stream()
                .map(g -> fmt(g.recallAtK())).collect(Collectors.joining(",")));
        String llmRecallData   = bpData(data.results(), r -> r.llmRuns().stream()
                .map(l -> fmt(l.recall())).collect(Collectors.joining(",")));
        String llmPrecData     = bpData(data.results(), r -> r.llmRuns().stream()
                .map(l -> fmt(l.precision())).collect(Collectors.joining(",")));
        String llmF1Data       = bpData(data.results(), r -> r.llmRuns().stream()
                .map(l -> fmt(l.f1())).collect(Collectors.joining(",")));

        return "<script>\n(function(){\n"
                + "const bpLabels=[" + labels + "];\n"
                + "const bpOpts={responsive:true,plugins:{legend:{position:'bottom'}},scales:{y:{beginAtZero:true,max:1.0}}};\n"
                + "new Chart(document.getElementById('bpGraph'),{type:'boxplot',data:{labels:bpLabels,datasets:[\n"
                + "  {label:'MRR',      data:[" + graphMrrData    + "],backgroundColor:'rgba(54,162,235,0.5)',borderColor:'rgba(54,162,235,1)'},\n"
                + "  {label:'NDCG@k',   data:[" + graphNdcgData   + "],backgroundColor:'rgba(54,162,235,0.3)',borderColor:'rgba(54,162,235,0.7)'},\n"
                + "  {label:'Recall@k', data:[" + graphRecallData + "],backgroundColor:'rgba(54,162,235,0.15)',borderColor:'rgba(54,162,235,0.5)'}\n"
                + "]},options:bpOpts});\n"
                + "new Chart(document.getElementById('bpLlm'),{type:'boxplot',data:{labels:bpLabels,datasets:[\n"
                + "  {label:'Recall',    data:[" + llmRecallData + "],backgroundColor:'rgba(255,159,64,0.5)',borderColor:'rgba(255,159,64,1)'},\n"
                + "  {label:'Precision', data:[" + llmPrecData   + "],backgroundColor:'rgba(75,192,192,0.5)',borderColor:'rgba(75,192,192,1)'},\n"
                + "  {label:'F1',        data:[" + llmF1Data     + "],backgroundColor:'rgba(153,102,255,0.5)',borderColor:'rgba(153,102,255,1)'}\n"
                + "]},options:bpOpts});\n"
                + "})();\n</script>\n";
    }

    private String bpData(List<AggregatedEvalResult> results,
                          java.util.function.Function<AggregatedEvalResult, String> fn) {
        return results.stream().map(r -> "[" + fn.apply(r) + "]").collect(Collectors.joining(","));
    }

    private String ds(String label, String data, String color) {
        return "      { label: '" + label + "', data: [" + data + "], backgroundColor: '" + color + "' },\n";
    }

    private String join(List<AggregatedEvalResult> list,
                        java.util.function.Function<AggregatedEvalResult, String> fn) {
        return list.stream().map(fn).collect(Collectors.joining(", "));
    }

    private String fmt(double v) { return String.format("%.2f", v); }

    private String css() {
        return """
                <style>
                  body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; background: #f9f9f9; color: #222; }
                  header { margin-bottom: 1.5rem; }
                  h1 { margin: 0 0 .25rem; }
                  .ts { color: #666; margin: 0 0 1rem; }
                  .cfg { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
                  .ci { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .5rem 1rem; }
                  .ci span { display: block; font-size: .75rem; color: #888; }
                  .cards { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
                  .card { flex: 1; min-width: 110px; border-radius: 8px; padding: .8rem 1rem; color: #fff; text-align: center; }
                  .card .lbl { font-size: .8rem; opacity: .9; }
                  .card .val { font-size: 1.8rem; font-weight: bold; }
                  .green { background: #2e7d32; } .yellow { background: #f57f17; } .red { background: #c62828; }
                  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
                  th, td { padding: .5rem .7rem; text-align: left; border-bottom: 1px solid #eee; }
                  thead { background: #333; color: #fff; }
                  td.m { color: #999; font-size: .85rem; }
                  tr.good { background: #e8f5e9; } tr.warn { background: #fff8e1; } tr.bad { background: #ffebee; }
                  tr.best-mode { background: #e8f5e9; font-weight: 600; }
                  details { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: .75rem; }
                  summary { padding: .6rem 1rem; cursor: pointer; }
                  .db { padding: .5rem 1rem 1rem; }
                  table.dt { width: auto; box-shadow: none; margin-bottom: 1rem; }
                  table.dt thead { background: #eee; }
                  table.dt th { color: #333; }
                  td.top { background: #c8e6c9; font-weight: bold; }
                  td.not-found { color: #aaa; }
                  pre.resp { background: #f5f5f5; border-left: 4px solid #ddd; padding: .75rem 1rem; white-space: pre-wrap; word-break: break-word; font-size: .85rem; }
                  .chart-wrap { max-width: 900px; margin-bottom: 2rem; }
                  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
                  .chart-grid .chart-wrap { max-width: 100%; }
                  @media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
                  .sub { color: #666; margin: -.5rem 0 1rem; font-size: .9rem; }
                  h2 { margin-top: 1.5rem; } h4 { margin: 1rem 0 .4rem; }
                  section { margin-bottom: .5rem; }
                </style>
                """;
    }

    private String cfgItem(String label, String value) {
        return "<div class=\"ci\"><span>" + escape(label) + "</span><strong>" + escape(value) + "</strong></div>\n";
    }

    private String card(String label, double value) {
        String cls = value >= 0.8 ? "green" : value >= 0.5 ? "yellow" : "red";
        return String.format("<div class=\"card %s\"><div class=\"lbl\">%s</div><div class=\"val\">%.2f</div></div>%n",
                cls, escape(label), value);
    }

    private String rc(double v) { return v >= 0.8 ? "good" : v >= 0.5 ? "warn" : "bad"; }
    private String emoji(double v) { return v >= 0.8 ? "✅" : v >= 0.5 ? "⚠️" : "❌"; }

    private String joinList(List<String> list) {
        if (list == null || list.isEmpty()) return "—";
        return String.join(", ", list);
    }

    private String escape(String t) {
        if (t == null) return "";
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;");
    }
}
