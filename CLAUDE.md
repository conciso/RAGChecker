# ragcheck

## Zweck
Java-Tool zur strukturierten Evaluation der LightRAG-Retrieval-Qualität.

Das übergeordnete System besteht aus:
- **Datensenke-MVP** (separates Repo): Spring Boot App, überwacht SFTP/FTP/Local-Verzeichnis auf PDFs und uploaded sie an LightRAG
- **LightRAG** REST API (http://lightrag:9622): Graph-basiertes RAG-System
- **ragcheck** (dieses Repo): Evaluations-Tool für die Retrieval-Qualität

## Was ragcheck tut
Vordefinierte Prompts werden gegen die LightRAG API gefeuert.
Die zurückgegebenen Dokumente werden mit einer Erwartungsliste verglichen.
Die Antwort des LLM ist irrelevant — ausschließlich die retrieved Documents werden bewertet.

## Metriken
- Recall: Anteil der erwarteten Dokumente, die tatsächlich retrieved wurden
- Precision: Anteil der retrieved Dokumente, die erwartet wurden
- F1-Score: Harmonisches Mittel

## Analysephasen
1. Baseline: Datenerreichbarkeit messen (Recall/Precision auf ~10 Testfällen)
2. Parameteroptimierung: z.B. Similarity Threshold, Datenstruktur, Prompts
3. Regressionssicherheit: Qualität bei neuen Dokumenten stabil halten?

## Testfall-Format (YAML)
```yaml
- id: TC-001
  prompt: "Wer ist für das Onboarding neuer Mitarbeiter zuständig?"
  expected_documents:
    - "onboarding_prozess_2024.pdf"
    - "hr_handbuch.pdf"
  notes: "Kernprozess HR"
```

## Tech Stack
- Java 21, Spring Boot 3.4.x, Maven
- groupId: de.conciso
- LightRAG API-Endpunkt konfigurierbar via .env / application.properties

## LightRAG API
- POST /query → Retrieval-Query
- POST /documents/paginated → Dokumentenliste abfragen
- Basis-URL per Umgebungsvariable konfigurieren (z.B. LIGHTRAG_URL)
