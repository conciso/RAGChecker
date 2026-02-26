package de.conciso.ragcheck.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class LightRagClient {

    private static final Logger log = LoggerFactory.getLogger(LightRagClient.class);

    // Matches "[N] some filename.pdf", optionally preceded by a markdown list marker ("* " or "- ")
    private static final Pattern REFERENCE_PATTERN =
            Pattern.compile("^[-*]?\\s*\\[(\\d+)]\\s*(.+?)\\s*$", Pattern.MULTILINE);

    private final RestClient restClient;
    private final String queryMode;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public LightRagClient(
            @Value("${ragchecker.lightrag.url}") String baseUrl,
            @Value("${ragchecker.lightrag.api-key}") String apiKey,
            @Value("${ragchecker.query.mode}") String queryMode
    ) {
        RestClient.Builder builder = RestClient.builder().baseUrl(baseUrl);
        if (apiKey != null && !apiKey.isBlank()) {
            builder.defaultHeader("X-API-Key", apiKey);
        }
        this.restClient = builder.build();
        this.queryMode = queryMode;
    }

    /**
     * Graph-Retrieval via /query/data.
     * Gibt die referenzierten Dokumente in Reihenfolge zurück (Position = Relevanz-Rang).
     * Kein LLM-Aufruf — schnell.
     */
    public List<String> queryData(String prompt) {
        QueryDataResponse response = restClient.post()
                .uri("/query/data")
                .body(Map.of(
                        "query", prompt,
                        "mode", queryMode,
                        "include_references", true
                ))
                .retrieve()
                .body(QueryDataResponse.class);

        if (response == null || response.data() == null) {
            log.warn("Empty /query/data response for: {}", prompt);
            return List.of();
        }

        QueryData data = response.data();

        List<String> fromReferences = data.references() == null ? List.of() :
                data.references().stream()
                        .map(Reference::filePath)
                        .filter(Objects::nonNull)
                        .toList();

        List<String> fromChunks = data.chunks() == null ? List.of() :
                data.chunks().stream()
                        .map(Chunk::filePath)
                        .filter(Objects::nonNull)
                        .toList();

        List<String> fromEntities = data.entities() == null ? List.of() :
                data.entities().stream()
                        .map(Entity::filePath)
                        .filter(Objects::nonNull)
                        .flatMap(p -> splitSep(p).stream())
                        .toList();

        List<String> fromRelationships = data.relationships() == null ? List.of() :
                data.relationships().stream()
                        .map(Relationship::filePath)
                        .filter(Objects::nonNull)
                        .flatMap(p -> splitSep(p).stream())
                        .toList();

        log.debug("filePaths from references    : {}", fromReferences);
        log.debug("filePaths from chunks        : {}", fromChunks);
        log.debug("filePaths from entities      : {}", fromEntities);
        log.debug("filePaths from relationships : {}", fromRelationships);

        // References are the primary ranked list; others are supplementary
        List<String> ordered = fromReferences.stream()
                .map(p -> Paths.get(p).getFileName().toString())
                .distinct()
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new));

        // Append any additional filenames not already present (from chunks/entities/relationships)
        java.util.stream.Stream.of(fromChunks, fromEntities, fromRelationships)
                .flatMap(List::stream)
                .map(p -> Paths.get(p).getFileName().toString())
                .filter(name -> !ordered.contains(name))
                .distinct()
                .forEach(ordered::add);

        log.debug("Ordered documents for ranking: {}", ordered);
        return List.copyOf(ordered);
    }

    /**
     * LLM-Aufruf via /query.
     * Läuft das vollständige RAG-Pipeline inkl. LLM — dauert 30–60 s.
     * Gibt den Antworttext und die darin zitierten Dokumente zurück.
     */
    public LlmResponse queryLlm(String prompt) {
        String raw = restClient.post()
                .uri("/query")
                .body(Map.of(
                        "query", prompt,
                        "mode", queryMode,
                        "stream", false
                ))
                .retrieve()
                .body(String.class);

        if (raw == null || raw.isBlank()) {
            log.warn("Empty /query response for: {}", prompt);
            return new LlmResponse(null, List.of());
        }

        log.debug("Raw /query response:\n{}", raw);

        String responseText = extractResponseText(raw);
        List<String> fileNames = parseReferencedFiles(responseText);

        log.debug("LLM referenced files: {}", fileNames);
        return new LlmResponse(responseText, fileNames);
    }

    public record LlmResponse(String responseText, List<String> referencedDocuments) {}

    // --- helpers ---

    private String extractResponseText(String raw) {
        String trimmed = raw.trim();
        if (trimmed.startsWith("{")) {
            try {
                JsonNode node = objectMapper.readTree(trimmed);
                for (String field : List.of("response", "data", "result", "answer")) {
                    if (node.has(field) && node.get(field).isTextual()) {
                        return node.get(field).asText();
                    }
                }
            } catch (Exception e) {
                log.debug("Response is not valid JSON, treating as plain text");
            }
        }
        return trimmed;
    }

    private List<String> parseReferencedFiles(String text) {
        if (text == null || text.isBlank()) return List.of();
        List<String> files = new ArrayList<>();
        Matcher matcher = REFERENCE_PATTERN.matcher(text);
        while (matcher.find()) {
            String ref = matcher.group(2).trim();
            if (!ref.isBlank()) files.add(ref);
        }
        return files.stream().distinct().toList();
    }

    /** Splits "<SEP>"-concatenated paths that LightRAG sometimes produces in entity file_path. */
    private List<String> splitSep(String path) {
        if (path == null) return List.of();
        return List.of(path.split("<SEP>"));
    }

    // --- /query/data response records ---

    @JsonIgnoreProperties(ignoreUnknown = true)
    record QueryDataResponse(QueryData data) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record QueryData(
            List<Entity> entities,
            List<Relationship> relationships,
            List<Chunk> chunks,
            List<Reference> references
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Entity(
            @JsonProperty("entity_name") String entityName,
            @JsonProperty("file_path") String filePath
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Relationship(
            @JsonProperty("src_id") String srcId,
            @JsonProperty("tgt_id") String tgtId,
            @JsonProperty("file_path") String filePath
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Chunk(
            @JsonProperty("file_path") String filePath,
            @JsonProperty("chunk_id") String chunkId
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Reference(
            @JsonProperty("reference_id") String referenceId,
            @JsonProperty("file_path") String filePath
    ) {}
}
