package de.conciso.ragcheck.report;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Liest bekannte LightRAG-Parameter direkt aus den Umgebungsvariablen
 * und baut daraus ein kompaktes Run-Label zusammen.
 */
@Component
public class OverrideEnvLoader {

    private static final Logger log = LoggerFactory.getLogger(OverrideEnvLoader.class);

    private static final List<String> PARAM_KEYS = List.of(
            "COSINE_THRESHOLD",
            "TOP_K",
            "CHUNK_TOP_K",
            "MAX_ENTITY_TOKENS",
            "MAX_RELATION_TOKENS",
            "RELATED_CHUNK_NUMBER",
            "KG_CHUNK_PICK_METHOD",
            "RERANK_BY_DEFAULT",
            "MIN_RERANK_SCORE"
    );

    /**
     * Liest alle bekannten LightRAG-Parameter aus den Umgebungsvariablen.
     * Liefert eine leere Map wenn keine der Variablen gesetzt ist.
     */
    public Map<String, String> load() {
        Map<String, String> params = new LinkedHashMap<>();
        for (String key : PARAM_KEYS) {
            String value = System.getenv(key);
            if (value != null && !value.isBlank()) {
                params.put(key, value.trim());
            }
        }
        if (params.isEmpty()) {
            log.debug("Keine LightRAG-Parameter-Env-Vars gefunden.");
        } else {
            log.info("LightRAG-Parameter aus Umgebungsvariablen geladen ({} Werte)", params.size());
        }
        return params;
    }

    /**
     * Baut aus den gesetzten Parametern ein lesbares Label zusammen,
     * z.B. "top_k=60, COSINE_THRESHOLD=0.4, CHUNK_TOP_K=21, …".
     * Liefert leeren String wenn keine Parameter gesetzt sind.
     */
    public String buildLabel() {
        Map<String, String> params = load();
        if (params.isEmpty()) return "";
        return params.entrySet().stream()
                .map(e -> displayKey(e.getKey()) + "=" + e.getValue())
                .collect(Collectors.joining(", "));
    }

    private static String displayKey(String key) {
        return "TOP_K".equals(key) ? "top_k" : key;
    }
}
