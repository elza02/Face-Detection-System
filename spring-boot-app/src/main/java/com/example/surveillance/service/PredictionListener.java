package com.example.surveillance.service;

import com.example.surveillance.config.ImageWebSocketHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

@Service
public class PredictionListener {

    @Autowired
    private ImageWebSocketHandler imageWebSocketHandler;

    @KafkaListener(topics = "prediction-topic", groupId = "group_id")
    public void listen(String message) {
        imageWebSocketHandler.sendPrediction(message); // Notify WebSocket clients
    }
}
