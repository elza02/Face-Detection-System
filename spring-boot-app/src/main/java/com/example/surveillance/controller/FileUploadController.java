package com.example.surveillance.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
public class FileUploadController {

    @Autowired
    private KafkaTemplate<String, byte[]> kafkaTemplate;

    private final String topic = "imageTest";

    @PostMapping("/uploadImage")
    public ResponseEntity<String> handleFileImageUpload(@RequestParam("file") MultipartFile file) throws IOException {
        byte[] bytes = file.getBytes();
        kafkaTemplate.send(topic, file.getOriginalFilename(), bytes);
        return ResponseEntity.ok("File uploaded and processed successfully");
    }
}