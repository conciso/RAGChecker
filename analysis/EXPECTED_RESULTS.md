# Erwartete Analyse-Ergebnisse (Fake-Daten, seed=42)

Referenz zum Abgleich nach Ausführung von `parameter_analysis.py --reports reports/fake`.

---

## Eingabe-Läufe (F1 ohne Rauschen → mit Rauschen)

| Label | query_mode | chunk_size | min_rerank | top_k | F1 erwartet | F1 tatsächlich |
|---|---|---|---|---|---|---|
| hybrid_512_03_15  | hybrid | 512  | 0.3 | 15 | 0.9700 | 0.9535 |
| hybrid_512_03_10  | hybrid | 512  | 0.3 | 10 | 0.9600 | 0.9656 |
| hybrid_512_03_20  | hybrid | 512  | 0.3 | 20 | 0.9300 | 0.9379 |
| hybrid_512_03_05  | hybrid | 512  | 0.3 |  5 | 0.9100 | 0.9160 |
| hybrid_512_05_10  | hybrid | 512  | 0.5 | 10 | 0.8500 | 0.8521 |
| hybrid_512_01_10  | hybrid | 512  | 0.1 | 10 | 0.8300 | 0.8439 |
| local_512_03_10   | local  | 512  | 0.3 | 10 | 0.7600 | 0.7674 |
| hybrid_1024_03_10 | hybrid | 1024 | 0.3 | 10 | 0.7600 | 0.7546 |
| hybrid_256_03_10  | hybrid | 256  | 0.3 | 10 | 0.7100 | 0.6991 |
| global_512_03_10  | global | 512  | 0.3 | 10 | 0.7100 | 0.7156 |
| local_512_05_15   | local  | 512  | 0.5 | 15 | 0.6600 | 0.6584 |
| local_512_01_10   | local  | 512  | 0.1 | 10 | 0.6300 | 0.6184 |
| hybrid_1024_05_20 | hybrid | 1024 | 0.5 | 20 | 0.6200 | 0.6068 |
| local_1024_03_10  | local  | 1024 | 0.3 | 10 | 0.5600 | 0.5651 |
| local_256_03_10   | local  | 256  | 0.3 | 10 | 0.5100 | 0.5060 |
| global_256_03_10  | global | 256  | 0.3 | 10 | 0.4600 | 0.4600 |
| global_1024_05_10 | global | 1024 | 0.5 | 10 | 0.4000 | 0.3861 |
| local_256_01_05   | local  | 256  | 0.1 |  5 | 0.3300 | 0.3488 |

---

## Phase 2 – SHAP

### Erwartetes Feature-Ranking (wichtigste zuerst)

| Rang | Feature | Eingebaute Effektgröße |
|---|---|---|
| 1 | `query_mode` | hybrid +0.25 vs. global 0.0 → Spanne **0.25** |
| 2 | `param_chunk_size` | 512 +0.15 vs. 256 −0.10 → Spanne **0.25** |
| 3 | `param_min_rerank_score` | 0.3 +0.08 vs. 0.1 −0.05 → Spanne **0.13** |
| 4 | `top_k` | 15 +0.04 vs. 5 −0.02 → Spanne **0.06** |

> `query_mode` und `param_chunk_size` haben dieselbe Gesamtspanne (0.25).
> Die genaue Reihenfolge dieser beiden kann je nach Modell variieren – beide deutlich vor `min_rerank_score`.

### Erwartete SHAP Dependence Plots

| Parameter | Erwartetes Muster |
|---|---|
| `query_mode` | hybrid → positive SHAP, global → negative SHAP |
| `param_chunk_size` | 512 → höchste SHAP, 256 → niedrigste SHAP, 1024 → leicht negativ |
| `param_min_rerank_score` | 0.3 → positiv, 0.1 und 0.5 → negativ |
| `top_k` | 10–15 → leicht positiv, 5 → leicht negativ, 20 → neutral |

---

## Phase 3 – Optuna

### Erwartete Top-1-Empfehlung

| Parameter | Erwarteter Wert |
|---|---|
| `query_mode` | `hybrid` |
| `param_chunk_size` | `512` |
| `param_min_rerank_score` | `0.3` |
| `top_k` | `15` |
| predicted `llm_f1` | **≈ 0.95–0.97** |

> Optuna sollte diese Kombination als beste noch nicht vollständig getestete Konfiguration vorschlagen
> (hybrid+512+0.3+15 ist zwar in den Daten, aber mit top_k=10 – top_k=15 bei sonst gleichen Params ebenfalls).

---

## Phase 4 – DoWhy

### Erwarteter Average Treatment Effect (ATE)

| Setting | Erwarteter Wert |
|---|---|
| Treatment (auto) | `query_mode` (binär: hybrid=1 vs. nicht-hybrid=0) |
| ATE | **+0.20 bis +0.25** |
| Interpretation | Wechsel zu `hybrid` erhöht `llm_f1` kausal um ~20 Punkte |

### Refutation-Tests

| Test | Erwartetes Ergebnis |
|---|---|
| Random Common Cause | Neuer Effekt bleibt nahe am Original ✓ |
| Placebo Treatment | Effekt verschwindet (nahe 0) ✓ |
| Data Subset | Effekt bleibt stabil ✓ |

### Counterfactuals (erste 5 Läufe)

Für alle `local`- und `global`-Läufe unter den ersten 5:
→ counterfactual `llm_f1` sollte **höher** sein als actual (delta ≈ +0.20)
Für `hybrid`-Läufe:
→ delta ≈ **−0.20** (wäre schlechter ohne hybrid)
