# RAGChecker – Parametereinfluss-Analyse: Konzept & Umsetzung

## Ziel

Die Pipeline führt RAGChecker mit variierenden LightRAG-Parametrisierungen mehrfach aus.
Jeder Lauf erzeugt eine `ragcheck_*.json` mit den genutzten Parametern und den Retrieval-Metriken.

Ziel dieser Analyse ist es:
1. **Welche Parameter** haben den größten Einfluss auf die Retrieval-Qualität?
2. **Welche Parameterkombinationen** führen zu den besten Ergebnissen?
3. **Was sollte als nächstes getestet werden?** (datengetriebene Empfehlungen)

---

## Datengrundlage

Jede `ragcheck_*.json` enthält:

```json
{
  "configuration": {
    "runLabel": "hybrid_chunk512",
    "runParameters": {
      "CHUNK_TOP_K": "10",
      "MIN_RERANK_SCORE": "0.3"
    },
    "queryMode": "hybrid",
    "topK": 10
  },
  "summary": {
    "llm": { "avgF1": 0.89, "avgRecall": 1.0, "avgPrecision": 0.83 },
    "graph": { "avgMrr": 0.5, "avgNdcgAtK": 0.63, "avgRecallAtK": 1.0 }
  }
}
```

**Features (Eingabe):** `queryMode`, `topK` + alle freien `runParameters` (z.B. `CHUNK_TOP_K`, `MIN_RERANK_SCORE`, `CHUNK_SIZE`, ...)

**Targets (Ausgabe):** `llm_f1` (primär), `llm_recall`, `llm_precision`, `graph_recall`, `graph_mrr`, ...

---

## Analyse-Methoden

### Phase 1 – Exploration (Korrelation & Verteilung)

**Methoden:**
- Korrelationsmatrix: Parameter ↔ Metriken (Pearson/Spearman)
- Boxplots: Wie verteilen sich die Metriken je Parameterwert? (z.B. queryMode = hybrid vs. local)
- Scatterplots: Numerische Parameter vs. Zielmetrik (mit Trendlinie)
- Übersichts-Balkendiagramm: Alle Läufe sortiert nach F1

**Eignung:** Ideal als erster Überblick, auch mit wenigen Datenpunkten (< 10 Läufe) sinnvoll.

**Output:** PNG-Plots in `analysis/output/`

---

### Phase 2 – SHAP Feature Importance + Dependence Plots

**Surrogate-Modell:** Random Forest (`--model rf`, default) oder XGBoost (`--model xgb`).
XGBoost passt bei mehr Daten oft besser, SHAP-Werte werden identisch berechnet.

**Idee:** Trainiere das Surrogate-Modell auf den Parametern → Metriken.
Nutze SHAP (SHapley Additive exPlanations) um den Beitrag jedes Parameters zu erklären.

**Vorteile:**
- **Global:** Welcher Parameter ist insgesamt der wichtigste?
- **Lokal:** Warum war Lauf X gut/schlecht?
- Funktioniert auch mit kleinen Datasets (10–30 Läufe) via `TreeExplainer`
- Zeigt nicht nur Richtung (positiv/negativ), sondern auch Magnitude

**SHAP Dependence Plots** (Kernplot dieser Phase):
- x-Achse = roher Parameterwert
- y-Achse = SHAP-Wert (Beitrag dieses Parameters zum Output)
- Farbe = automatisch erkannte stärkste Interaktion mit anderem Feature

→ Direkt ablesbar: in welchem Wertebereich trägt ein Parameter positiv bei?
  z.B. „`top_k` zwischen 5–15 hat positive SHAP-Werte, darüber fällt er ab"

**Partial Dependence Plots (PDP)** (ergänzend):
Zeigt den marginalen Effekt auf den *vorhergesagten Score* (ceteris paribus) —
nützlich für Schwellwerte und Sättigungseffekte.

```
Beispielausgabe SHAP Importance:

CHUNK_SIZE         ████████████████████ 0.42  ← wichtigster Parameter
MIN_RERANK_SCORE   █████████████        0.28
query_mode         ████████             0.18
top_k              █████                0.12
```

**Einschränkung:** Mit < 5 Datenpunkten ist das Modell überangepasst.
Ab 15+ Läufen ist die Aussagekraft gut, ab 30+ sehr gut.

**Output:** SHAP Summary (Beeswarm), SHAP Importance (Bar), SHAP Dependence Plots (2c),
PDP (2d, ergänzend), optional Interaktionsmatrix (2e, ab 10 Läufen)

---

### Phase 3 – Bayesian Optimization (Optuna Surrogate-Modell)

**Idee:** Optuna lernt aus vergangenen Läufen ein Surrogate-Modell (TPE: Tree-structured
Parzen Estimator) und schlägt die *nächsten* vielversprechenden Parameterkombinationen vor.

**Workflow:**
1. Vergangene Läufe als abgeschlossene Trials in eine Optuna-Studie laden
2. Optuna inferiert den Parameter-Raum (Bounds aus Min/Max, Kategorien aus beobachteten Werten)
3. Surrogate-Modell wird auf Basis der Trials gefittet
4. Top-N neue Parameterkombinationen werden vorgeschlagen (nicht getestet, nur empfohlen)

**Ausgabe:**
- Tabelle mit N vorgeschlagenen Parameterkombinationen + erwartetem Score
- Optuna FANOVA-Importance (unabhängige Bestätigung von Phase 2)
- Optimierungshistorie (wie schnell konvergiert das Modell?)

```
Empfehlung 1 (predicted llm_f1: 0.91):
  CHUNK_SIZE         = 512
  MIN_RERANK_SCORE   = 0.25
  query_mode         = hybrid
  top_k              = 15

Empfehlung 2 (predicted llm_f1: 0.88):
  ...
```

**Vorteil gegenüber Grid Search:** Konzentriert sich auf vielversprechende Regionen
des Parameterraums, nicht auf vollständige Abdeckung.

---

### Phase 4 – Kausale Analyse (DoWhy)

**Idee:** Unterscheidet Korrelation von kausalem Effekt.
Beantwortet: „Hat Parameter X einen *kausalen* Effekt auf den Score, oder ist es nur Korrelation?"

**Workflow:**
1. DAG definieren (auto-generiert: alle Parameter → Outcome, keine Confoundings zwischen Parametern)
2. Treatment-Feature wählen (`--treatment param_rerank` oder auto-Auswahl)
3. `identify_effect()` + `estimate_effect()` → **ATE** (Average Treatment Effect)
4. `refute_estimate()` → Sensitivitätschecks (Random Confounder, Placebo Treatment, Data Subset)
5. **Counterfactuals**: „Was wäre der Score für Lauf X, wenn Treatment=1 statt 0?"

**Beispielinterpretation:**
- ATE = +0.12 → Reranking erhöht F1 kausal um ~12 Punkte
- Placebo-Refutation bestätigt: Effekt verschwindet bei zufälligem Treatment ✓

**Einschränkungen:**
- Erfordert mind. 6 Läufe; ab 20 Läufen deutlich zuverlässiger
- DAG-Annahme (keine Confoundings) muss zum echten System passen
- Bei wenigen Daten nur als Richtungsindikator, nicht als Beweis verwenden

**Output:** ATE-Wert, Refutation-Checks (Konsole), Counterfactual-CSV, ATE-Plot

## Nicht implementiert

### LIME (Local Interpretable Model-agnostic Explanations)

Erklärt einzelne Predictions lokal. SHAP ist für unser Use Case (globale Wichtigkeit)
besser geeignet und interpretierbar. SHAP wurde bewusst bevorzugt.

---

## Technischer Stack

```
Python 3.10+
pandas          – Datenladen & DataFrame
numpy           – Numerik
matplotlib      – Plots
seaborn         – Heatmaps & statistische Visualisierungen
scikit-learn    – Random Forest, Partial Dependence Plots
xgboost         – XGBoost als alternatives Surrogate-Modell (--model xgb)
shap            – SHAP Explainer, Dependence Plots, Interaktionen
optuna          – Bayesian Optimization & FANOVA-Importance
dowhy           – Kausale Analyse: ATE, Counterfactuals, Refutation
```

---

## Verzeichnisstruktur

```
analysis/
├── VORSCHLAG.md              ← Dieses Dokument
├── parameter_analysis.py     ← Hauptscript (alle 4 Phasen)
├── requirements.txt          ← Python-Abhängigkeiten
└── output/                   ← Generierte Plots & CSV (gitignore)
    ├── 1a_korrelation.png
    ├── 1b_boxplot_query_mode.png
    ├── 1c_scatter_params.png
    ├── 1d_laeufe_uebersicht.png
    ├── 2a_shap_summary.png
    ├── 2b_shap_importance.png
    ├── 2c_shap_dependence.png       ← SHAP Dependence (x=Wert, y=SHAP, Farbe=Interaktion)
    ├── 2d_partial_dependence.png    ← PDP ergänzend (x=Wert, y=predicted Score)
    ├── 2e_shap_interaktion.png      ← ab 10 Läufen
    ├── 3a_optuna_importance.png
    ├── 3b_optuna_history.png
    ├── 3_optuna_suggestions.csv
    ├── 4a_dowhy_ate.png
    └── 4_counterfactuals.csv
```

---

## Nutzung

```bash
# Setup
cd analysis/
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Alle Phasen (Reports-Verzeichnis relativ zu analysis/)
python parameter_analysis.py --reports ../reports

# Nur eine Phase ausführen
python parameter_analysis.py --reports ../reports --phase 2

# XGBoost als Surrogate statt Random Forest
python parameter_analysis.py --reports ../reports --model xgb

# Andere Zielmetrik (default: llm_f1)
python parameter_analysis.py --reports ../reports --target graph_recall

# Mehr Optuna-Empfehlungen
python parameter_analysis.py --reports ../reports --suggestions 10

# DoWhy: explizites Treatment-Feature angeben
python parameter_analysis.py --reports ../reports --phase 4 --treatment query_mode

# Eigener Reports-Pfad (z.B. reports nach run1/, run2/, ...)
python parameter_analysis.py --reports /pfad/zu/reports/
```

---

## Hinweise zur Datenqualität

| Anzahl Läufe | Aussagekraft                                           |
|:------------:|--------------------------------------------------------|
| < 5          | Nur Exploration sinnvoll; ML-Phasen sehr eingeschränkt |
| 5–15         | SHAP-Trends erkennbar, Optuna gibt erste Richtungen    |
| 15–30        | Gute Aussagekraft, Cross-Validation möglich            |
| 30+          | Sehr gute Aussagekraft, Interaktionseffekte erkennbar  |

**Empfehlung:** Mindestens 2 Werte pro Parameter variieren, Parameter möglichst
unabhängig voneinander variieren (One-Factor-at-a-Time oder Latin Hypercube Sampling).

---

## Bekannte Einschränkungen & geplante Erweiterungen

- `runParameters` sind freitextlich → unterschiedliche Keys über Läufe hinweg werden
  als sparse Features behandelt (fehlende Werte = Median)
- Ordner-Struktur `reports/run1/`, `reports/run2/` etc. wird bereits unterstützt
  (rekursiver Glob)
- Geplant: Per-Testfall-Analyse (nicht nur Summary-Metriken)
- Geplant: Kausalanalyse mit DoWhy sobald > 50 Läufe vorhanden
