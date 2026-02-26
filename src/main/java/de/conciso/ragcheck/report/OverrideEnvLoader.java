package de.conciso.ragcheck.report;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Liest /config/override.env (oder konfigurierten Pfad) und liefert
 * die dort gesetzten LightRAG-Parameter als Map für den Report.
 *
 * Erwartetes Format: KEY=VALUE, eine Zeile pro Eintrag.
 * Leerzeilen und Zeilen mit '#' werden ignoriert.
 */
@Component
public class OverrideEnvLoader {

    private static final Logger log = LoggerFactory.getLogger(OverrideEnvLoader.class);

    private final String overrideEnvPath;

    public OverrideEnvLoader(
            @Value("${ragchecker.override-env.path:/config/override.env}") String overrideEnvPath
    ) {
        this.overrideEnvPath = overrideEnvPath;
    }

    /**
     * Gibt alle KEY=VALUE-Paare aus der override.env zurück.
     * Liefert eine leere Map wenn die Datei nicht existiert oder nicht lesbar ist.
     */
    public Map<String, String> load() {
        Path path = Path.of(overrideEnvPath);
        if (!Files.exists(path)) {
            log.debug("override.env nicht gefunden: {}", path);
            return Map.of();
        }

        Map<String, String> params = new LinkedHashMap<>();
        try {
            for (String line : Files.readAllLines(path)) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;
                int eq = trimmed.indexOf('=');
                if (eq > 0) {
                    String key = trimmed.substring(0, eq).trim();
                    String value = trimmed.substring(eq + 1).trim();
                    params.put(key, value);
                }
            }
            log.info("override.env geladen ({} Parameter): {}", params.size(), path);
        } catch (IOException e) {
            log.warn("override.env konnte nicht gelesen werden: {}", path, e);
        }
        return params;
    }
}
