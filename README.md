# RAGChecker

Evaluations-Tool zur strukturierten Messung der Retrieval-Qualität von [LightRAG](https://github.com/HKUDS/LightRAG).

Vordefinierte Testfälle werden gegen die LightRAG-API gefeuert. Die zurückgegebenen Dokumente werden mit einer Erwartungsliste verglichen. Die LLM-Antwort selbst ist irrelevant — ausschließlich die *retrieved Documents* werden bewertet.

---

## Metriken

| Metrik | Beschreibung |
|---|---|
| **Recall** | Anteil der erwarteten Dokumente, die tatsächlich retrieved wurden |
| **Precision** | Anteil der retrieved Dokumente, die erwartet wurden |
| **F1-Score** | Harmonisches Mittel aus Recall und Precision |

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
# .env anpassen (LightRAG-URL, Testfall-Pfad, …)

# 2. Bauen
mvn package -DskipTests

# 3. Ausführen
java -jar target/*.jar
```

Nach dem Lauf erscheinen im `reports/`-Verzeichnis drei Dateien:

```
reports/
  ragcheck_20260220_153045.json
  ragcheck_20260220_153045.md
  ragcheck_20260220_153045.html
```

---

## Konfiguration

Alle Parameter können per `.env`-Datei oder Umgebungsvariable gesetzt werden.

| Umgebungsvariable | Standard | Beschreibung |
|---|---|---|
| `RAGCHECKER_LIGHTRAG_URL` | `http://localhost:9622` | Basis-URL der LightRAG API |
| `RAGCHECKER_LIGHTRAG_API_KEY` | *(leer)* | Bearer-Token, falls die API abgesichert ist |
| `RAGCHECKER_QUERY_MODE` | `hybrid` | Retrieval-Modus: `local` \| `global` \| `hybrid` \| `naive` \| `mix` |
| `RAGCHECKER_TOP_K` | `10` | Anzahl der zurückgegebenen Top-Entitäten/Relationen |
| `RAGCHECKER_TESTCASES_PATH` | `testcases/testcases.yaml` | Pfad zur Testfall-YAML-Datei |
| `RAGCHECKER_OUTPUT_PATH` | `reports` | Ausgabeverzeichnis für Reports |

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

Nach jedem Lauf werden drei Dateien mit identischem Zeitstempel erzeugt.

### JSON

Maschinenlesbares Format für CI-Auswertung oder Trendvergleiche.

```json
{
  "timestamp": "2026-02-20T15:30:45Z",
  "configuration": { "queryMode": "hybrid", "topK": 10, "testCasesPath": "..." },
  "summary": { "avgRecall": 0.8, "avgPrecision": 0.7, "avgF1": 0.75, "totalTestCases": 12 },
  "results": [
    {
      "id": "TC-001",
      "prompt": "...",
      "recall": 1.0, "precision": 0.5, "f1": 0.67,
      "expectedDocuments": ["Fall1_Abmahnung_15_05_25.pdf"],
      "retrievedDocuments": ["Fall1_Abmahnung_15_05_25.pdf", "Fall1_Klage.pdf"],
      "entityCount": 5, "relationshipCount": 3, "chunkCount": 2,
      "highLevelKeywords": ["Arbeitsrecht"], "lowLevelKeywords": ["Abmahnung"]
    }
  ]
}
```

### Markdown

Git-/Confluence-freundliches Format. Konfiguration, Zusammenfassung und Ergebnistabelle mit Emoji-Status:

- ✅ F1 ≥ 0.8
- ⚠️ F1 ≥ 0.5
- ❌ F1 < 0.5

### HTML

Selbstständige HTML-Datei (kein lokaler Server nötig). Enthält:

- Farbige Summary-Cards (grün / gelb / rot je nach Schwellwert)
- Farbkodierte Ergebnistabelle
- Grouped-Bar-Diagramm (Recall / Precision / F1 je Testfall) via [Chart.js](https://www.chartjs.org/)
- Aufklappbare Detail-Abschnitte mit erwarteten/gefundenen Dokumenten und Keywords

---

## Docker

```bash
# Bauen
docker build -t ragcheck .

# Ausführen (mit externem aibox_network)
docker compose up
```

`docker-compose.yml` bindet `./testcases` als Read-only-Volume ein und setzt Umgebungsvariablen aus `.env`.

Für lokale Tests ohne externes Netzwerk liegt eine `docker-compose-local.yml` bereit.

---

## Projektstruktur

```
.
├── src/main/java/de/conciso/ragcheck/
│   ├── RagCheckerApplication.java       # Spring Boot Entry Point
│   ├── api/
│   │   └── LightRagClient.java          # REST-Client für LightRAG
│   ├── model/
│   │   ├── EvalResult.java              # Evaluationsergebnis (Record)
│   │   ├── QueryResult.java             # LightRAG-Antwort (Record)
│   │   └── TestCase.java                # Testfall-Definition (Record)
│   ├── report/
│   │   ├── ReportData.java              # Alle Daten eines Laufs (Record)
│   │   ├── ReportWriter.java            # Orchestriert JSON/MD/HTML
│   │   ├── JsonReportWriter.java        # Schreibt .json
│   │   ├── MarkdownReportWriter.java    # Schreibt .md
│   │   └── HtmlReportWriter.java        # Schreibt .html mit Chart.js
│   ├── runner/
│   │   └── EvaluationRunner.java        # CommandLineRunner
│   └── service/
│       ├── EvaluationService.java       # Evaluationslogik
│       └── TestCaseLoader.java          # YAML-Loader
├── testcases/
│   └── fall1.yaml                       # Beispiel: Arbeitsrecht, 12 Fälle
├── reports/                             # Generierte Reports (gitignore empfohlen)
├── .env.example                         # Konfigurationsvorlage
├── docker-compose.yml
└── Dockerfile
```

---

## Analysephasen

1. **Baseline** — Datenerreichbarkeit messen: Recall/Precision auf ~10 Testfällen
2. **Parameteroptimierung** — z.B. Query Mode, Top-K, Prompts variieren und JSON-Reports vergleichen
3. **Regressionssicherheit** — nach neuen Dokumenten prüfen, ob die Qualität stabil bleibt

---

## Tech Stack

- Java 21, Spring Boot 3.4.x, Maven
- Jackson (JSON + YAML)
- LightRAG REST API (`POST /query/data`)
