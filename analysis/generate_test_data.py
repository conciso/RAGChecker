#!/usr/bin/env python3
"""
RAGChecker Test-Datengenerator
================================

Generiert synthetische ragcheck_*.json Dateien mit bekannten Parametereffekten,
sodass die Ausgabe von parameter_analysis.py vorhersagbar und prüfbar ist.

Verwendung:
  python generate_test_data.py                        # erzeugt reports/fake/
  python generate_test_data.py --out reports/myfake   # eigenes Verzeichnis

Nach der Generierung einfach ausführen:
  python parameter_analysis.py --reports reports/fake

Erwartete Ergebnisse sind am Ende dieser Datei dokumentiert
und werden beim Ausführen auch auf der Konsole ausgegeben.
"""

import json
import argparse
import random
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# BEKANNTE PARAMETEREFFEKTE (Ground Truth)
# ─────────────────────────────────────────────────────────────
#
# llm_f1 = BASE
#         + MODE_BONUS[query_mode]
#         + CHUNK_BONUS[chunk_size]
#         + RERANK_BONUS[min_rerank_score]
#         + TOPK_BONUS[top_k]
#         + Rauschen (±0.02, seed=42)
#
# Erwartetes SHAP-Ranking (wichtigste zuerst):
#   1. query_mode        (Effektgröße: +0.25 für hybrid vs. 0.0 für global → größte Spanne)
#   2. param_chunk_size  (Effektgröße: +0.15 für 512 vs. -0.10 für 256)
#   3. param_min_rerank_score (Effektgröße: +0.08 für 0.3 vs. -0.05 für 0.1)
#   4. top_k             (Effektgröße: +0.04 für 15 vs. -0.02 für 5 → kleinste Spanne)
#
# Erwarteter bester Lauf:
#   query_mode=hybrid, chunk_size=512, min_rerank_score=0.3, top_k=15
#   → theoretisches F1 = 0.45 + 0.25 + 0.15 + 0.08 + 0.04 = 0.97

BASE = 0.45

MODE_BONUS = {
    "hybrid": 0.25,   # bester Modus
    "local":  0.05,   # mittelmäßig
    "global": 0.00,   # schlechtester Modus
}

CHUNK_BONUS = {
    256:  -0.10,  # zu klein → schlechter Kontext
    512:  +0.15,  # optimal
    1024: -0.05,  # zu groß → Rauschen nimmt zu
}

RERANK_BONUS = {
    0.1: -0.05,  # zu niedrig → zu viel Rauschen durch
    0.3: +0.08,  # optimal
    0.5: -0.03,  # zu streng → relevante Docs gefiltert
}

TOPK_BONUS = {
    5:  -0.02,  # zu wenig Kandidaten
    10: +0.03,  # gut
    15: +0.04,  # leicht besser
    20:  0.00,  # zu viele → kein Gewinn mehr
}

# ─────────────────────────────────────────────────────────────
# 18 DEFINIERTE LÄUFE (alle Kombinationen die wir testen)
# ─────────────────────────────────────────────────────────────

RUNS = [
    # (label,              query_mode, chunk_size, min_rerank, top_k)
    ("hybrid_512_03_10",  "hybrid",  512,  0.3, 10),  # guter Referenzlauf
    ("hybrid_512_03_15",  "hybrid",  512,  0.3, 15),  # bester Lauf erwartet
    ("hybrid_512_03_05",  "hybrid",  512,  0.3,  5),  # top_k zu niedrig
    ("hybrid_512_03_20",  "hybrid",  512,  0.3, 20),  # top_k zu hoch
    ("hybrid_512_01_10",  "hybrid",  512,  0.1, 10),  # rerank zu niedrig
    ("hybrid_512_05_10",  "hybrid",  512,  0.5, 10),  # rerank zu hoch
    ("hybrid_256_03_10",  "hybrid",  256,  0.3, 10),  # chunk zu klein
    ("hybrid_1024_03_10", "hybrid", 1024,  0.3, 10),  # chunk zu groß
    ("hybrid_1024_05_20", "hybrid", 1024,  0.5, 20),  # schlechtester hybrid
    ("local_512_03_10",   "local",   512,  0.3, 10),  # local referenz
    ("local_512_01_10",   "local",   512,  0.1, 10),  # local, schlechter rerank
    ("local_512_05_15",   "local",   512,  0.5, 15),  # local, strenger rerank
    ("local_256_03_10",   "local",   256,  0.3, 10),  # local, kleine chunks
    ("local_1024_03_10",  "local",  1024,  0.3, 10),  # local, große chunks
    ("local_256_01_05",   "local",   256,  0.1,  5),  # schlechtester local
    ("global_512_03_10",  "global",  512,  0.3, 10),  # global referenz
    ("global_256_03_10",  "global",  256,  0.3, 10),  # global, kleine chunks
    ("global_1024_05_10", "global", 1024,  0.5, 10),  # schlechtester global
]


def compute_expected_f1(query_mode: str, chunk_size: int, min_rerank: float, top_k: int) -> float:
    return (
        BASE
        + MODE_BONUS[query_mode]
        + CHUNK_BONUS[chunk_size]
        + RERANK_BONUS[min_rerank]
        + TOPK_BONUS[top_k]
    )


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def build_metrics(f1: float, rng: random.Random) -> dict:
    """Leitet alle Metriken aus dem F1-Wert ab (mit kleinem Rauschen)."""
    f1_noisy = _clip(f1 + rng.uniform(-0.02, 0.02), 0.05, 1.0)

    recall    = _clip(f1_noisy + rng.uniform(-0.05, 0.10))
    precision = _clip(2 * f1_noisy * recall / max(f1_noisy + recall, 1e-9))
    mrr       = _clip(f1_noisy + rng.uniform(-0.08, 0.05))
    hitrate   = _clip(f1_noisy + rng.uniform(0.0, 0.15))

    graph_recall = _clip(recall + rng.uniform(-0.05, 0.08))
    graph_mrr    = _clip(mrr    + rng.uniform(-0.10, 0.05))
    graph_ndcg   = _clip(f1_noisy + rng.uniform(-0.06, 0.06))

    return {
        "llm_f1": round(f1_noisy, 4),
        "llm_recall": round(recall, 4),
        "llm_precision": round(precision, 4),
        "llm_mrr": round(mrr, 4),
        "llm_hitrate": round(hitrate, 4),
        "graph_recall": round(graph_recall, 4),
        "graph_mrr": round(graph_mrr, 4),
        "graph_ndcg": round(graph_ndcg, 4),
    }


def build_json(
    label: str,
    query_mode: str,
    chunk_size: int,
    min_rerank: float,
    top_k: int,
    metrics: dict,
    run_index: int,
) -> dict:
    ts = (datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc) + timedelta(hours=run_index)).isoformat()
    return {
        "timestamp": ts,
        "configuration": {
            "runLabel": label,
            "runParameters": {
                "CHUNK_SIZE": str(chunk_size),
                "MIN_RERANK_SCORE": str(min_rerank),
            },
            "queryMode": query_mode,
            "topK": top_k,
            "runsPerTestCase": 3,
            "testCasesPath": "testcases/fake.yaml",
        },
        "summary": {
            "graph": {
                "avgMrr": metrics["graph_mrr"],
                "avgNdcgAtK": metrics["graph_ndcg"],
                "avgRecallAtK": metrics["graph_recall"],
                "avgDurationMs": int(1500 + run_index * 37),
            },
            "llm": {
                "avgRecall": metrics["llm_recall"],
                "avgPrecision": metrics["llm_precision"],
                "avgF1": metrics["llm_f1"],
                "avgHitRate": metrics["llm_hitrate"],
                "avgMrr": metrics["llm_mrr"],
                "avgDurationMs": int(4800 + run_index * 113),
            },
            "totalTestCases": 3,
        },
        "results": [
            {
                "id": f"TC-00{i+1}",
                "prompt": f"Testfrage {i+1}",
                "expectedDocuments": [f"doc_{i+1}.pdf"],
                "runs": 3,
                "graph": {
                    "avgMrr": round(metrics["graph_mrr"] + (i - 1) * 0.03, 4),
                    "stdDevMrr": 0.02,
                    "avgNdcgAtK": round(metrics["graph_ndcg"] + (i - 1) * 0.02, 4),
                    "stdDevNdcgAtK": 0.02,
                    "avgRecallAtK": round(metrics["graph_recall"] + (i - 1) * 0.02, 4),
                    "stdDevRecallAtK": 0.02,
                    "avgDurationMs": 1500,
                    "ranksByDoc": {f"doc_{i+1}.pdf": [1, 1, 2]},
                    "runs": [{"retrievedDocuments": [f"doc_{i+1}.pdf"], "mrr": metrics["graph_mrr"], "ndcgAtK": metrics["graph_ndcg"], "recallAtK": metrics["graph_recall"], "durationMs": 1500}],
                },
                "llm": {
                    "avgRecall": round(metrics["llm_recall"] + (i - 1) * 0.02, 4),
                    "stdDevRecall": 0.03,
                    "avgPrecision": round(metrics["llm_precision"] + (i - 1) * 0.02, 4),
                    "stdDevPrecision": 0.03,
                    "avgF1": round(metrics["llm_f1"] + (i - 1) * 0.02, 4),
                    "stdDevF1": 0.03,
                    "hitRate": metrics["llm_hitrate"],
                    "avgMrr": metrics["llm_mrr"],
                    "stdDevMrr": 0.03,
                    "avgDurationMs": 4800,
                    "runs": [{"retrievedDocuments": [f"doc_{i+1}.pdf"], "recall": metrics["llm_recall"], "precision": metrics["llm_precision"], "f1": metrics["llm_f1"], "hit": True, "mrr": metrics["llm_mrr"], "durationMs": 4800, "responseText": "Fake response."}],
                },
            }
            for i in range(3)
        ],
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generiert synthetische ragcheck-Testdaten")
    parser.add_argument("--out", type=Path, default=Path("reports/fake"),
                        help="Ausgabeverzeichnis (default: reports/fake)")
    parser.add_argument("--seed", type=int, default=42, help="Zufalls-Seed (default: 42)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("RAGChecker Test-Datengenerator")
    print("=" * 60)
    print(f"Ausgabe: {args.out.resolve()}")
    print(f"Seed:    {args.seed}\n")

    # ── Dateien erzeugen ──────────────────────────────────────
    generated = []
    for i, (label, mode, chunk, rerank, topk) in enumerate(RUNS):
        expected_f1 = compute_expected_f1(mode, chunk, rerank, topk)
        metrics = build_metrics(expected_f1, rng)
        data = build_json(label, mode, chunk, rerank, topk, metrics, i)

        ts_str = data["timestamp"].replace(":", "").replace("-", "").replace("T", "_")[:15]
        filename = args.out / f"ragcheck_{ts_str}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        generated.append((label, mode, chunk, rerank, topk, expected_f1, metrics["llm_f1"]))

    # ── Erwartete Ergebnisse ausgeben ─────────────────────────
    print("Generierte Läufe (sortiert nach erwartetem F1):")
    print("-" * 90)
    print(f"{'Label':<25} {'Mode':<8} {'Chunk':>6} {'Rerank':>7} {'TopK':>5}  {'F1 erwartet':>12}  {'F1 tatsächl.':>13}")
    print("-" * 90)

    for row in sorted(generated, key=lambda r: r[5], reverse=True):
        label, mode, chunk, rerank, topk, exp_f1, act_f1 = row
        marker = " ← BESTER" if exp_f1 == max(r[5] for r in generated) else ""
        print(f"{label:<25} {mode:<8} {chunk:>6} {rerank:>7.1f} {topk:>5}  {exp_f1:>12.4f}  {act_f1:>13.4f}{marker}")

    print("\n")
    print("=" * 60)
    print("ERWARTETE ANALYSE-ERGEBNISSE")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│  SHAP Ranking (Phase 2) – erwartete Reihenfolge:        │
│                                                         │
│  1. query_mode          Spanne: 0.25  (hybrid vs global)│
│  2. param_chunk_size    Spanne: 0.25  (512 vs 256)      │
│  3. param_min_rerank_score  Spanne: 0.13  (0.3 vs 0.1)  │
│  4. top_k               Spanne: 0.06  (15 vs 5)         │
│                                                         │
│  SHAP Dependence Plots – erwartete Muster:              │
│  • query_mode=hybrid    → positive SHAP-Werte           │
│  • query_mode=global    → negative SHAP-Werte           │
│  • chunk_size=512       → höchste SHAP-Werte            │
│  • chunk_size=256       → niedrigste SHAP-Werte         │
│  • min_rerank=0.3       → positive SHAP-Werte           │
│  • top_k 10–15          → leicht positive SHAP-Werte    │
├─────────────────────────────────────────────────────────┤
│  Optuna (Phase 3) – erwartete Top-Empfehlung:           │
│                                                         │
│  query_mode=hybrid, chunk_size=512,                     │
│  min_rerank_score=0.3,  top_k=15                        │
│  predicted F1 ≈ 0.97                                    │
├─────────────────────────────────────────────────────────┤
│  DoWhy (Phase 4) – erwarteter ATE:                      │
│                                                         │
│  Treatment: query_mode (hybrid=1 vs. nicht-hybrid=0)    │
│  ATE ≈ +0.20 bis +0.25                                  │
│  → Wechsel zu hybrid erhöht F1 kausal um ~20 Punkte     │
│  Refutation: Effekt sollte bei Placebo verschwinden ✓   │
└─────────────────────────────────────────────────────────┘
""")

    print(f"✅  {len(RUNS)} JSON-Dateien erzeugt in: {args.out.resolve()}")
    print("\nJetzt ausführen:")
    print(f"  python parameter_analysis.py --reports {args.out}")


if __name__ == "__main__":
    main()
