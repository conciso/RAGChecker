# RAGChecker

Evaluations-Tool zur strukturierten Messung der Retrieval-Qualität von [LightRAG](https://github.com/HKUDS/LightRAG).

Pro Testfall werden **beide** LightRAG-Endpunkte aufgerufen und getrennt bewertet:

| Endpunkt | Was wird gemessen |
|---|---|
| `POST /query/data` | Graph-Retrieval — wie gut findet der Knowledge Graph relevante Dokumente, und an welcher Position? |
| `POST /query` | LLM-Antwort — welche Dokumente zitiert das LLM tatsächlich in seiner Antwort? |

Jeder Testfall wird N-mal ausgeführt (konfigurierbar), um den Nicht-Determinismus der Keyword-Generierung statistisch auszugleichen.

---

## Metriken

### Graph-Retrieval (`/query/data`)

| Metrik | Beschreibung |
|---|---|
| **Recall@k** | Anteil der erwarteten Dokumente, die unter den Top-k Retrieved sind |
| **MRR** | Mean Reciprocal Rank — wie weit oben steht das erste relevante Dokument? |
| **NDCG@k** | Normalized Discounted Cumulative Gain — positions-gewichtete Qualität |
| **Rang je Dokument** | Exakte Position jedes erwarteten Dokuments in der Ergebnisliste |

### LLM (`/query`)

| Metrik | Beschreibung |
|---|---|
| **Recall** | Anteil der erwarteten Dokumente, die das LLM zitiert hat |
| **Precision** | Anteil der zitierten Dokumente, die erwartet wurden |
| **F1-Score** | Harmonisches Mittel aus Recall und Precision |
| **Hit-Rate** | Anteil der Läufe, in denen mind. ein erwartetes Dokument zitiert wurde |
| **MRR** | Mean Reciprocal Rank über die LLM-Zitate |

Alle Metriken werden als **Mittelwert ± Standardabweichung** über N Läufe ausgegeben.

---

## Voraussetzungen

- Java 21+
- Maven 3.9+
- LightRAG REST API erreichbar (Standard: `http://localhost:9622`)

---

## Schnellstart

```bash
# 1. Konfiguration anlegen
cp .env.example .env
# .env anpassen (LightRAG-URL, Testfall-Pfad, Anzahl Läufe, …)

# 2. Bauen
mvn package -DskipTests

# 3. Ausführen
java -jar target/*.jar
```

Nach dem Lauf erscheinen im `reports/`-Verzeichnis drei Dateien:

```
reports/
  ragcheck_20260223_153045.json
  ragcheck_20260223_153045.md
  ragcheck_20260223_153045.html
```

---

## Konfiguration

Alle Parameter können per `.env`-Datei oder Umgebungsvariable gesetzt werden.

### Grundkonfiguration

| Umgebungsvariable | Standard | Beschreibung |
|---|---|---|
| `RAGCHECKER_LIGHTRAG_URL` | `http://localhost:9622` | Basis-URL der LightRAG API |
| `RAGCHECKER_LIGHTRAG_API_KEY` | *(leer)* | Bearer-Token, falls die API abgesichert ist |
| `RAGCHECKER_QUERY_MODE` | `hybrid` | Retrieval-Modus: `local` \| `global` \| `hybrid` \| `naive` \| `mix` |
| `RAGCHECKER_TOP_K` | `10` | Anzahl Top-Entitäten/Relationen (k für Ranking-Metriken) |
| `RAGCHECKER_RUNS_PER_TESTCASE` | `1` | Anzahl Läufe je Testfall (empfohlen: 3–5) |
| `RAGCHECKER_TESTCASES_PATH` | `testcases/testcases.yaml` | Pfad zur Testfall-YAML-Datei |
| `RAGCHECKER_OUTPUT_PATH` | `reports` | Ausgabeverzeichnis für Reports |

### Modus & Parametervergleich

| Umgebungsvariable | Standard | Beschreibung |
|---|---|---|
| `RAGCHECKER_MODE` | `evaluate` | Betriebsmodus: `evaluate` \| `compare` |
| `RAGCHECKER_RUN_LABEL` | *(leer)* | Kurzer Name für diesen Parametersatz, z.B. `hybrid_chunk512` |
| `RAGCHECKER_RUN_PARAMETERS` | *(leer)* | Beliebige LightRAG-Parameter als `key=value`-Paare, kommagetrennt |

`RAGCHECKER_RUN_PARAMETERS` dient der Dokumentation der LightRAG-internen Parameter, die von der Pipeline gesetzt wurden (z.B. `chunk_size=512,similarity_threshold=0.7`). Diese Werte werden in den Report geschrieben und beim Vergleich angezeigt — ragcheck selbst wertet sie nicht aus.

---

## Testfälle

Testfälle werden als YAML-Datei definiert:

```yaml
- id: TC-001
  prompt: "Wer ist für das Onboarding neuer Mitarbeiter zuständig?"
  expected_documents:
    - "onboarding_prozess_2024.pdf"
    - "hr_handbuch.pdf"
  notes: "Kernprozess HR"
```

| Feld | Pflicht | Beschreibung |
|---|---|---|
| `id` | ja | Eindeutige Test-ID, erscheint in allen Reports |
| `prompt` | ja | Frage, die an LightRAG gesendet wird |
| `expected_documents` | ja | Dateinamen der Dokumente, die retrieved werden sollen |
| `notes` | nein | Freitext-Notiz, wird nicht ausgewertet |

Ein Treffer gilt, wenn ein retrieved Dateiname den erwarteten Dateinamen enthält (case-insensitiv).

Beispiel-Testfälle liegen unter [`testcases/fall1.yaml`](testcases/fall1.yaml) (Arbeitsrecht, 12 Fälle).

---

## Reports

### Einzellauf (`RAGCHECKER_MODE=evaluate`)

Nach jedem Lauf werden drei Dateien mit identischem Zeitstempel erzeugt.

#### JSON

Maschinenlesbares Format für CI-Auswertung und Läufe-Vergleiche. Enthält je Testfall getrennte `graph`- und `llm`-Sektionen mit aggregierten Metriken und Rohdaten je Lauf. Alle Run-Metadaten (`runLabel`, `runParameters`) sind im `configuration`-Block gespeichert und werden vom Comparison-Modus ausgelesen.

```json
{
  "timestamp": "...",
  "configuration": {
    "runLabel": "hybrid_chunk512",
    "runParameters": { "chunk_size": "512", "similarity_threshold": "0.7" },
    "queryMode": "hybrid",
    "topK": 10,
    "runsPerTestCase": 3,
    "testCasesPath": "..."
  },
  "summary": {
    "graph": { "avgMrr": 0.75, "avgNdcgAtK": 0.68, "avgRecallAtK": 0.80 },
    "llm":   { "avgRecall": 0.80, "avgPrecision": 0.70, "avgF1": 0.75, "avgHitRate": 0.90, "avgMrr": 0.65 }
  },
  "results": [
    {
      "id": "TC-001", "prompt": "...", "expectedDocuments": [...], "runs": 3,
      "graph": {
        "avgMrr": 0.5, "stdDevMrr": 0.0,
        "ranksByDoc": { "dokument.pdf": [2, 2, 2] },
        "runs": [{ "retrievedDocuments": [...], "mrr": 0.5, "ndcgAtK": 0.63, "recallAtK": 1.0 }]
      },
      "llm": {
        "avgRecall": 1.0, "avgPrecision": 1.0, "avgF1": 1.0, "hitRate": 1.0,
        "runs": [{ "retrievedDocuments": [...], "recall": 1.0, "f1": 1.0, "hit": true, "responseText": "..." }]
      }
    }
  ]
}
```

#### Markdown

Git-/Confluence-freundliches Format mit zwei getrennten Ergebnistabellen (Graph / LLM), Rang-Matrix je Testfall und Emoji-Status:

- ✅ Metrik ≥ 0.8
- ⚠️ Metrik ≥ 0.5
- ❌ Metrik < 0.5

#### HTML

Selbstständige HTML-Datei (kein lokaler Server nötig). Enthält:

- Summary-Cards für Graph (MRR, NDCG@k, Recall@k) und LLM (Recall, Precision, F1, Hit-Rate, MRR)
- Farbkodierte Ergebnistabellen für beide Endpunkte
- Grouped-Bar-Diagramm (alle Metriken je Testfall) via [Chart.js](https://www.chartjs.org/)
- Aufklappbare Detail-Abschnitte mit Rang-Matrix, Per-Lauf-Tabellen und LLM-Antworttext

### Parametervergleich (`RAGCHECKER_MODE=compare`)

Liest alle `ragcheck_*.json`-Dateien im Output-Verzeichnis und erzeugt einen Vergleichsreport:

```
reports/
  comparison.md
  comparison.html
```

Der Report zeigt alle Läufe nebeneinander, sortiert nach Ø LLM-F1 (absteigend). Der beste Wert je Spalte ist markiert (★ im Markdown, farblich im HTML). Die HTML-Tabellen sind per Klick auf den Spalten-Header sortierbar.

---

## Parameteroptimierung über eine Pipeline

Typischer Einsatz: LightRAG wird mit verschiedenen Parametersätzen neu gestartet, jeweils gefolgt von einem ragcheck-Lauf. Am Ende erstellt der Comparison-Modus eine Übersicht aller Ergebnisse.

```bash
#!/usr/bin/env bash
set -e

REPORT_DIR="reports/run_$(date +%Y%m%d_%H%M%S)"

run_evaluation() {
  local label="$1"
  local params="$2"
  local mode="$3"
  local topk="$4"

  echo "=== Starte Lauf: $label ==="

  # LightRAG mit Parametern starten (projektspezifisch)
  # docker compose up -d lightrag

  # ragcheck ausführen
  RAGCHECKER_MODE=evaluate \
  RAGCHECKER_RUN_LABEL="$label" \
  RAGCHECKER_RUN_PARAMETERS="$params" \
  RAGCHECKER_QUERY_MODE="$mode" \
  RAGCHECKER_TOP_K="$topk" \
  RAGCHECKER_OUTPUT_PATH="$REPORT_DIR" \
  java -jar target/ragcheck.jar

  # LightRAG stoppen
  # docker compose stop lightrag
}

# Parametersätze durchlaufen
run_evaluation "hybrid_top10"  "chunk_size=512,similarity_threshold=0.7"  "hybrid"  10
run_evaluation "hybrid_top20"  "chunk_size=512,similarity_threshold=0.7"  "hybrid"  20
run_evaluation "local_top10"   "chunk_size=256,similarity_threshold=0.5"  "local"   10

# Vergleichsreport erzeugen
echo "=== Erstelle Vergleichsreport ==="
RAGCHECKER_MODE=compare \
RAGCHECKER_OUTPUT_PATH="$REPORT_DIR" \
java -jar target/ragcheck.jar

echo "Fertig: $REPORT_DIR/comparison.html"
```

---

## Docker

```bash
# Produktivbetrieb (erfordert externes aibox_network)
docker compose up --build

# Lokaler Test gegen LightRAG auf dem Host
docker compose -f docker-compose-local.yml up --build
```

`docker-compose-local.yml` nutzt `host.docker.internal` statt des Service-Namens und benötigt kein externes Netzwerk. Reports werden in `./reports` auf dem Host gemountet.

---

## Projektstruktur

```
.
├── src/main/java/de/conciso/ragcheck/
│   ├── RagCheckerApplication.java           # Spring Boot Entry Point
│   ├── api/
│   │   └── LightRagClient.java              # queryData() + queryLlm()
│   ├── model/
│   │   ├── TestCase.java                    # Testfall-Definition (Record)
│   │   ├── GraphRetrievalRunResult.java     # Ergebnis eines Graph-Laufs
│   │   ├── LlmRunResult.java                # Ergebnis eines LLM-Laufs
│   │   ├── AggregatedGraphMetrics.java      # Aggregat über N Graph-Läufe
│   │   ├── AggregatedLlmMetrics.java        # Aggregat über N LLM-Läufe
│   │   └── AggregatedEvalResult.java        # Kombiniertes Ergebnis je Testfall
│   ├── report/
│   │   ├── ReportData.java                  # Alle Daten eines Laufs (inkl. Run-Metadaten)
│   │   ├── ReportWriter.java                # Orchestriert JSON/MD/HTML
│   │   ├── JsonReportWriter.java            # Schreibt .json
│   │   ├── MarkdownReportWriter.java        # Schreibt .md
│   │   ├── HtmlReportWriter.java            # Schreibt .html mit Chart.js
│   │   └── ComparisonReportWriter.java      # Schreibt comparison.md + comparison.html
│   ├── runner/
│   │   ├── EvaluationRunner.java            # Aktiv bei MODE=evaluate (Standard)
│   │   └── ComparisonRunner.java            # Aktiv bei MODE=compare
│   └── service/
│       ├── EvaluationService.java           # Evaluationslogik (N Läufe je Testfall)
│       └── TestCaseLoader.java              # YAML-Loader
├── testcases/
│   └── fall1.yaml                           # Beispiel: Arbeitsrecht, 12 Fälle
├── reports/                                 # Generierte Reports (gitignore empfohlen)
├── .env                                     # Lokale Konfiguration (nicht einchecken)
├── docker-compose.yml                       # Produktiv (aibox_network)
├── docker-compose-local.yml                 # Lokal (host.docker.internal)
└── Dockerfile
```

---

## Analysephasen

1. **Baseline** — Graph- und LLM-Qualität auf definierten Testfällen messen
2. **Parameteroptimierung** — LightRAG mit verschiedenen Parametern starten, je Lauf einen Report erzeugen, am Ende mit `MODE=compare` vergleichen
3. **Regressionssicherheit** — nach neuen Dokumenten prüfen, ob die Qualität mit dem besten Parametersatz stabil bleibt

---

## Tech Stack

- Java 21, Spring Boot 3.4.x, Maven
- Jackson (JSON + YAML)
- LightRAG REST API (`POST /query/data`, `POST /query`)
