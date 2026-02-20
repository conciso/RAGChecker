package de.conciso.ragcheck.model;

import java.util.List;

public record QueryResult(
        List<String> referencedFilePaths,
        List<String> referencedFileNames,
        List<String> entityNames,
        int entityCount,
        int relationshipCount,
        int chunkCount,
        List<String> highLevelKeywords,
        List<String> lowLevelKeywords
) {}
