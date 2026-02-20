package de.conciso.ragcheck.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import de.conciso.ragcheck.model.QueryResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

@Component
public class LightRagClient {

    private static final Logger log = LoggerFactory.getLogger(LightRagClient.class);

    private final RestClient restClient;
    private final String queryMode;
    private final int topK;

    public LightRagClient(
            @Value("${ragchecker.lightrag.url}") String baseUrl,
            @Value("${ragchecker.lightrag.api-key}") String apiKey,
            @Value("${ragchecker.query.mode}") String queryMode,
            @Value("${ragchecker.query.top-k}") int topK
    ) {
        RestClient.Builder builder = RestClient.builder().baseUrl(baseUrl);
        if (apiKey != null && !apiKey.isBlank()) {
            builder.defaultHeader("Authorization", "Bearer " + apiKey);
        }
        this.restClient = builder.build();
        this.queryMode = queryMode;
        this.topK = topK;
    }

    public QueryResult query(String prompt) {
        QueryDataResponse response = restClient.post()
                .uri("/query/data")
                .body(Map.of(
                        "query", prompt,
                        "mode", queryMode,
                        "include_references", true,
                        "top_k", topK
                ))
                .retrieve()
                .body(QueryDataResponse.class);

        if (response == null || response.data() == null) {
            log.warn("Empty response from LightRAG for prompt: {}", prompt);
            return new QueryResult(List.of(), List.of(), List.of(), 0, 0, 0, List.of(), List.of());
        }

        QueryData data = response.data();

        List<String> rawPaths = collectFilePaths(data);
        List<String> deduplicatedPaths = rawPaths.stream().distinct().toList();
        List<String> fileNames = deduplicatedPaths.stream()
                .map(p -> Paths.get(p).getFileName().toString())
                .distinct()
                .toList();

        List<String> entityNames = data.entities() == null ? List.of() :
                data.entities().stream()
                        .map(Entity::entityName)
                        .filter(Objects::nonNull)
                        .toList();

        int entityCount = data.entities() == null ? 0 : data.entities().size();
        int relationshipCount = data.relationships() == null ? 0 : data.relationships().size();
        int chunkCount = data.chunks() == null ? 0 : data.chunks().size();

        Metadata meta = response.metadata();
        List<String> highLevel = List.of();
        List<String> lowLevel = List.of();
        if (meta != null && meta.keywords() != null) {
            if (meta.keywords().highLevel() != null) highLevel = meta.keywords().highLevel();
            if (meta.keywords().lowLevel() != null) lowLevel = meta.keywords().lowLevel();
        }

        return new QueryResult(deduplicatedPaths, fileNames, entityNames,
                entityCount, relationshipCount, chunkCount, highLevel, lowLevel);
    }

    private List<String> collectFilePaths(QueryData data) {
        Stream<String> fromReferences = data.references() == null ? Stream.of() :
                data.references().stream()
                        .map(Reference::filePath)
                        .filter(Objects::nonNull);

        Stream<String> fromChunks = data.chunks() == null ? Stream.of() :
                data.chunks().stream()
                        .map(Chunk::filePath)
                        .filter(Objects::nonNull);

        Stream<String> fromEntities = data.entities() == null ? Stream.of() :
                data.entities().stream()
                        .map(Entity::filePath)
                        .filter(Objects::nonNull);

        Stream<String> fromRelationships = data.relationships() == null ? Stream.of() :
                data.relationships().stream()
                        .map(Relationship::filePath)
                        .filter(Objects::nonNull);

        return Stream.of(fromReferences, fromChunks, fromEntities, fromRelationships)
                .flatMap(s -> s)
                .toList();
    }

    // --- Response Records ---

    @JsonIgnoreProperties(ignoreUnknown = true)
    record QueryDataResponse(QueryData data, Metadata metadata) {}

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
            @JsonProperty("entity_type") String entityType,
            String description,
            @JsonProperty("file_path") String filePath
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Relationship(
            @JsonProperty("src_id") String srcId,
            @JsonProperty("tgt_id") String tgtId,
            String description,
            Double weight,
            @JsonProperty("file_path") String filePath
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Chunk(
            String content,
            @JsonProperty("file_path") String filePath,
            @JsonProperty("chunk_id") String chunkId
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Reference(
            @JsonProperty("reference_id") String referenceId,
            @JsonProperty("file_path") String filePath
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Metadata(
            @JsonProperty("query_mode") String queryMode,
            Keywords keywords,
            @JsonProperty("processing_info") ProcessingInfo processingInfo
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Keywords(
            @JsonProperty("high_level") List<String> highLevel,
            @JsonProperty("low_level") List<String> lowLevel
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record ProcessingInfo(
            @JsonProperty("total_entities_found") Integer totalEntitiesFound,
            @JsonProperty("total_relations_found") Integer totalRelationsFound,
            @JsonProperty("final_chunks_count") Integer finalChunksCount
    ) {}
}
