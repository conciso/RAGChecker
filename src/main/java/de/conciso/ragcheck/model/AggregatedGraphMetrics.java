package de.conciso.ragcheck.model;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public record AggregatedGraphMetrics(
        double avgMrr,       double stdDevMrr,
        double avgNdcgAtK,   double stdDevNdcgAtK,
        double avgRecallAtK, double stdDevRecallAtK,
        Map<String, List<Integer>> ranksByDoc
) {
    public static AggregatedGraphMetrics of(List<String> expectedDocs,
                                            List<GraphRetrievalRunResult> runs) {
        double[] mrrs    = runs.stream().mapToDouble(GraphRetrievalRunResult::mrr).toArray();
        double[] ndcgs   = runs.stream().mapToDouble(GraphRetrievalRunResult::ndcgAtK).toArray();
        double[] recalls = runs.stream().mapToDouble(GraphRetrievalRunResult::recallAtK).toArray();

        Map<String, List<Integer>> ranksByDoc = expectedDocs.stream().collect(
                Collectors.toMap(
                        doc -> doc,
                        doc -> runs.stream()
                                .map(r -> r.expectedDocumentRanks().getOrDefault(doc, 0))
                                .toList()
                )
        );

        return new AggregatedGraphMetrics(
                mean(mrrs),    stdDev(mrrs),
                mean(ndcgs),   stdDev(ndcgs),
                mean(recalls), stdDev(recalls),
                ranksByDoc
        );
    }

    private static double mean(double[] v) {
        if (v.length == 0) return 0.0;
        double s = 0;
        for (double x : v) s += x;
        return s / v.length;
    }

    private static double stdDev(double[] v) {
        if (v.length < 2) return 0.0;
        double avg = mean(v);
        double sq = 0;
        for (double x : v) sq += (x - avg) * (x - avg);
        return Math.sqrt(sq / v.length);
    }
}
