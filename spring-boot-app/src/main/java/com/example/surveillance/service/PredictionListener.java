package com.example.surveillance.service;

import com.example.surveillance.WebSocketHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class PredictionListener {

    private static final Logger logger = LoggerFactory.getLogger(PredictionListener.class);

    @Autowired
    private WebSocketHandler webSocketHandler;

    @KafkaListener(topics = "prediction-topic", groupId = "group_id")
    public void listen(String message) {
        logger.info("Received prediction from Kafka: {}", message);
        webSocketHandler.sendPrediction(message);
    }
}