# RAGChecker Parameter-Analyse – Umsetzungszusammenfassung

## Was wurde gebaut

`analysis/parameter_analysis.py` – ein Python-Script das alle `ragcheck_*.json` Läufe einliest und in 4 Phasen auswertet.

---

## Die 4 Phasen

| Phase | Was | CLI |
|---|---|---|
| 1 – Exploration | Korrelationsmatrix, Boxplots, Scatterplots | `--phase 1` |
| 2 – SHAP | Feature Importance + Dependence Plots + PDP | `--phase 2` |
| 3 – Optuna | Nächste Parameterkombinationen empfehlen | `--phase 3` |
| 4 – DoWhy | Kausale Analyse (ATE, Counterfactuals) | `--phase 4` |

---

## Abgleich mit dem Kollegen-Vorschlag

| Punkt aus dem Gespräch | Umgesetzt? | Anmerkung |
|---|---|---|
| Multi-Run-Table als Basis | ✅ | Alle `ragcheck_*.json` rekursiv → DataFrame |
| RF/XGBoost als Surrogate | ✅ | `--model rf` (default) oder `--model xgb` |
| SHAP TreeExplainer | ✅ | Phase 2 |
| Barplot globale Importance | ✅ | `2b_shap_importance.png` |
| **SHAP Dependence Plots** (x=Wert, y=SHAP) | ✅ | `2c_shap_dependence.png` – war im ersten Entwurf noch falsch (PDP statt SHAP Dependence), nach Vergleich korrigiert |
| SHAP Interaktionen (chunk × top_k) | ✅ | `2e_shap_interaktion.png` (ab 10 Läufen) |
| Konkrete Parameterempfehlungen | ✅ | Phase 3, CSV-Export |
| **DoWhy ATE + Refutation** | ✅ | Phase 4 – ATE, 3 Refutation-Tests |
| **Counterfactuals** | ✅ | Phase 4 – „Was wäre F1, wenn Treatment geflipt?" |

---

## Schnellstart

```bash
cd analysis/
pip install -r requirements.txt
python parameter_analysis.py --reports ../reports
```
