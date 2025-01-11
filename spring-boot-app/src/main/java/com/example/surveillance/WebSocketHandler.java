package com.example.surveillance;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.AbstractWebSocketHandler;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.nio.ByteBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Component
public class WebSocketHandler extends AbstractWebSocketHandler {

    private static final Logger logger = LoggerFactory.getLogger(WebSocketHandler.class);
    private final List<WebSocketSession> sessions = new ArrayList<>();
    
    @Autowired
    private KafkaTemplate<String, byte[]> kafkaTemplate;

    @Value("${kafka.topic.image:imageTest}")
    private String imageTopic;

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        sessions.add(session);
        logger.info("WebSocket connection established. Session ID: {}", session.getId());
    }

    @Override
    protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
        try {
            ByteBuffer buffer = message.getPayload();
            byte[] imageData = new byte[buffer.remaining()];
            buffer.get(imageData);
            
            logger.info("Received binary message of size: {} bytes from session: {}", 
                       imageData.length, session.getId());
            
            kafkaTemplate.send(imageTopic, imageData)
                .thenAccept(result -> {
                    logger.info("Successfully sent image to Kafka topic: {}", imageTopic);
                    try {
                        session.sendMessage(new TextMessage("{\"status\":\"processing\"}"));
                    } catch (IOException e) {
                        logger.error("Error sending processing status message", e);
                    }
                })
                .exceptionally(ex -> {
                    logger.error("Failed to send image to Kafka topic: {}", imageTopic, ex);
                    try {
                        session.sendMessage(new TextMessage("{\"error\":\"Failed to process image\"}"));
                    } catch (IOException e) {
                        logger.error("Error sending error message", e);
                    }
                    return null;
                });
        } catch (Exception e) {
            logger.error("Error processing binary message", e);
            sendErrorMessage(session, "Error processing image: " + e.getMessage());
        }
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) {
        logger.error("Transport error for session {}: {}", session.getId(), exception.getMessage());
        sessions.remove(session);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        sessions.remove(session);
        logger.info("WebSocket connection closed. Session ID: {} with status: {}", 
                   session.getId(), status);
    }

    public void sendPrediction(String prediction) {
        logger.info("Broadcasting prediction to {} connected sessions", sessions.size());
        for (WebSocketSession session : sessions) {
            if (session.isOpen()) {
                try {
                    session.sendMessage(new TextMessage(prediction));
                    logger.debug("Sent prediction to session: {}", session.getId());
                } catch (IOException e) {
                    logger.error("Error sending prediction to session: {}", session.getId(), e);
                }
            }
        }
    }

    private void sendErrorMessage(WebSocketSession session, String errorMessage) {
        try {
            String jsonError = String.format("{\"status\":\"error\",\"message\":\"%s\"}", errorMessage);
            session.sendMessage(new TextMessage(jsonError));
            logger.error("Sent error message to session {}: {}", session.getId(), errorMessage);
        } catch (IOException ex) {
            logger.error("Failed to send error message to session: {}", session.getId(), ex);
        }
    }
}