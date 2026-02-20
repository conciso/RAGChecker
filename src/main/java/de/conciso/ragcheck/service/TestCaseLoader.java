package de.conciso.ragcheck.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.conciso.ragcheck.model.TestCase;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Service
public class TestCaseLoader {

    private final String testCasesPath;
    private final ObjectMapper yaml = new ObjectMapper(new YAMLFactory());

    public TestCaseLoader(@Value("${ragchecker.testcases.path}") String testCasesPath) {
        this.testCasesPath = testCasesPath;
    }

    public List<TestCase> load() throws IOException {
        File file = new File(testCasesPath);
        return yaml.readValue(
                file,
                yaml.getTypeFactory().constructCollectionType(List.class, TestCase.class)
        );
    }
}
