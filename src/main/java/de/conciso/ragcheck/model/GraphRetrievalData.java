package de.conciso.ragcheck.model;

import java.util.List;
import java.util.Map;

/**
 * Ergebnis eines /query/data-Aufrufs.
 * Enthält neben den retrived Dokumenten (geordnet) auch die semantischen
 * Informationen, die LightRAG aus dem Wissensgraph extrahiert hat.
 */
public record GraphRetrievalData(
        /** Alle retrieved Dokumente in Relevanz-Reihenfolge (dedupliziert). */
        List<String> retrievedDocuments,
        /** Entity-Namen aus dem High-level-Kontext (Wissensgraph). */
        List<String> entityNames,
        /** Relationship-Paare im Format "A → B". */
        List<String> relationshipPairs,
        /** Dateinamen je Quelltyp: references, chunks, entities, relationships. */
        Map<String, List<String>> documentsBySource
) {}
