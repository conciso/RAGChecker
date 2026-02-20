package de.conciso.ragcheck.model;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public record TestCase(
        String id,
        String prompt,
        @JsonProperty("expected_documents") List<String> expectedDocuments,
        String notes
) {}
