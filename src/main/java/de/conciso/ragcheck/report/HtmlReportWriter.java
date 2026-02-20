package de.conciso.ragcheck.report;

import de.conciso.ragcheck.model.EvalResult;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

@Component
public class HtmlReportWriter {

    private static final DateTimeFormatter DISPLAY_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    public void write(ReportData data, Path path) throws IOException {
        String ts = DISPLAY_FORMAT.format(data.timestamp());
        String html = buildHtml(data, ts);
        Files.writeString(path, html, StandardCharsets.UTF_8);
    }

    private String buildHtml(ReportData data, String ts) {
        StringBuilder sb = new StringBuilder();

        sb.append("<!DOCTYPE html>\n<html lang=\"de\">\n<head>\n");
        sb.append("<meta charset=\"UTF-8\">\n");
        sb.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        sb.append("<title>RAGChecker Report — ").append(escape(ts)).append("</title>\n");
        sb.append("<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n");
        sb.append(inlineCss());
        sb.append("</head>\n<body>\n");

        // Header
        sb.append("<header>\n");
        sb.append("<h1>RAGChecker Report</h1>\n");
        sb.append("<p class=\"timestamp\">").append(escape(ts)).append("</p>\n");
        sb.append("<div class=\"config-grid\">\n");
        sb.append(configItem("Query Mode", data.queryMode()));
        sb.append(configItem("Top-K", String.valueOf(data.topK())));
        sb.append(configItem("Testfälle", data.testCasesPath()));
        sb.append("</div>\n</header>\n");

        // Summary cards
        sb.append("<section class=\"summary\">\n");
        sb.append(metricCard("Ø Recall", data.avgRecall()));
        sb.append(metricCard("Ø Precision", data.avgPrecision()));
        sb.append(metricCard("Ø F1", data.avgF1()));
        sb.append("</section>\n");

        // Results table
        sb.append("<section>\n<h2>Ergebnisse</h2>\n");
        sb.append("<table>\n<thead>\n<tr>");
        sb.append("<th>ID</th><th>Status</th><th>Recall</th><th>Precision</th><th>F1</th>");
        sb.append("<th>Retrieved</th><th>Expected</th>");
        sb.append("</tr>\n</thead>\n<tbody>\n");
        for (EvalResult r : data.results()) {
            String rowClass = rowClass(r.f1());
            String statusEmoji = statusEmoji(r.f1());
            sb.append(String.format(
                    "<tr class=\"%s\"><td>%s</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%d</td><td>%d</td></tr>%n",
                    rowClass, escape(r.testCaseId()), statusEmoji,
                    r.recall(), r.precision(), r.f1(),
                    r.retrievedDocuments().size(), r.expectedDocuments().size()));
        }
        sb.append("</tbody>\n</table>\n</section>\n");

        // Chart
        sb.append("<section>\n<h2>Metriken je Testfall</h2>\n");
        sb.append("<div class=\"chart-container\">\n");
        sb.append("<canvas id=\"metricsChart\"></canvas>\n");
        sb.append("</div>\n</section>\n");

        // Details
        sb.append("<section>\n<h2>Details</h2>\n");
        for (EvalResult r : data.results()) {
            sb.append("<details>\n");
            sb.append("<summary><strong>").append(escape(r.testCaseId())).append("</strong> — ")
                    .append(statusEmoji(r.f1()))
                    .append(String.format(" F1: %.2f</summary>%n", r.f1()));
            sb.append("<div class=\"detail-body\">\n");
            sb.append("<p><strong>Prompt:</strong> ").append(escape(r.prompt())).append("</p>\n");

            sb.append("<table class=\"detail-table\">\n");
            sb.append("<tr><th>Erwartet</th><td>").append(escape(joinList(r.expectedDocuments()))).append("</td></tr>\n");
            sb.append("<tr><th>Gefunden</th><td>").append(escape(joinList(r.retrievedDocuments()))).append("</td></tr>\n");
            sb.append("</table>\n");

            if (r.queryResult() != null) {
                List<String> high = r.queryResult().highLevelKeywords();
                List<String> low = r.queryResult().lowLevelKeywords();
                if (high != null && !high.isEmpty()) {
                    sb.append("<p><strong>Keywords (high):</strong> ").append(escape(String.join(", ", high))).append("</p>\n");
                }
                if (low != null && !low.isEmpty()) {
                    sb.append("<p><strong>Keywords (low):</strong> ").append(escape(String.join(", ", low))).append("</p>\n");
                }
            }

            sb.append("</div>\n</details>\n");
        }
        sb.append("</section>\n");

        // Chart.js script
        sb.append(buildChartScript(data));

        sb.append("</body>\n</html>\n");
        return sb.toString();
    }

    private String buildChartScript(ReportData data) {
        String labels = data.results().stream()
                .map(r -> "\"" + r.testCaseId().replace("\"", "\\\"") + "\"")
                .collect(Collectors.joining(", "));
        String recalls = data.results().stream()
                .map(r -> String.format("%.2f", r.recall()))
                .collect(Collectors.joining(", "));
        String precisions = data.results().stream()
                .map(r -> String.format("%.2f", r.precision()))
                .collect(Collectors.joining(", "));
        String f1s = data.results().stream()
                .map(r -> String.format("%.2f", r.f1()))
                .collect(Collectors.joining(", "));

        return """
                <script>
                  const ctx = document.getElementById('metricsChart').getContext('2d');
                  new Chart(ctx, {
                    type: 'bar',
                    data: {
                      labels: [""" + labels + """
                ],
                      datasets: [
                        {
                          label: 'Recall',
                          data: [""" + recalls + """
                ],
                          backgroundColor: 'rgba(54, 162, 235, 0.7)'
                        },
                        {
                          label: 'Precision',
                          data: [""" + precisions + """
                ],
                          backgroundColor: 'rgba(255, 159, 64, 0.7)'
                        },
                        {
                          label: 'F1',
                          data: [""" + f1s + """
                ],
                          backgroundColor: 'rgba(75, 192, 192, 0.7)'
                        }
                      ]
                    },
                    options: {
                      responsive: true,
                      scales: {
                        y: { beginAtZero: true, max: 1.0 }
                      }
                    }
                  });
                </script>
                """;
    }

    private String inlineCss() {
        return """
                <style>
                  body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; background: #f9f9f9; color: #222; }
                  header { margin-bottom: 1.5rem; }
                  h1 { margin: 0 0 0.25rem; }
                  .timestamp { color: #666; margin: 0 0 1rem; }
                  .config-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; }
                  .config-item { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem 1rem; }
                  .config-item span { display: block; font-size: 0.75rem; color: #888; }
                  .config-item strong { font-size: 1rem; }
                  .summary { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
                  .card { flex: 1; min-width: 140px; border-radius: 8px; padding: 1rem 1.5rem; color: #fff; text-align: center; }
                  .card .label { font-size: 0.85rem; opacity: 0.9; }
                  .card .value { font-size: 2rem; font-weight: bold; }
                  .card.green { background: #2e7d32; }
                  .card.yellow { background: #f57f17; }
                  .card.red { background: #c62828; }
                  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                  th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #eee; }
                  thead { background: #333; color: #fff; }
                  tr.good { background: #e8f5e9; }
                  tr.warn { background: #fff8e1; }
                  tr.bad  { background: #ffebee; }
                  details { background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 0.75rem; }
                  summary { padding: 0.6rem 1rem; cursor: pointer; }
                  .detail-body { padding: 0.5rem 1rem 1rem; }
                  .detail-table { width: auto; box-shadow: none; margin-bottom: 0.5rem; }
                  .detail-table th { background: #f0f0f0; color: #333; width: 8rem; }
                  .chart-container { max-width: 900px; margin-bottom: 2rem; }
                  h2 { margin-top: 2rem; }
                  section { margin-bottom: 1rem; }
                </style>
                """;
    }

    private String configItem(String label, String value) {
        return "<div class=\"config-item\"><span>" + escape(label) + "</span><strong>" + escape(value) + "</strong></div>\n";
    }

    private String metricCard(String label, double value) {
        String cssClass = value >= 0.8 ? "green" : value >= 0.5 ? "yellow" : "red";
        return String.format(
                "<div class=\"card %s\"><div class=\"label\">%s</div><div class=\"value\">%.2f</div></div>%n",
                cssClass, escape(label), value);
    }

    private String rowClass(double f1) {
        if (f1 >= 0.8) return "good";
        if (f1 >= 0.5) return "warn";
        return "bad";
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

    private String escape(String text) {
        if (text == null) return "";
        return text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("\"", "&quot;");
    }
}
