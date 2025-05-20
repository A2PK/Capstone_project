package gateway

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/IBM/sarama"
	"github.com/gofiber/websocket/v2"
	"github.com/google/uuid"
)

// WaterQualityData represents the structure of Kafka messages
type WaterQualityData struct {
	MonitoringTime  time.Time `json:"monitoring_time"`
	StationID       uuid.UUID `json:"station_id"`
	StationName     string    `json:"station_name"`
	Source          string    `json:"source"`
	ObservationType string    `json:"observation_type"`
	Features        []struct {
		Name  string   `json:"name"`
		Value *float64 `json:"value"`
	} `json:"features"`
}

// WebSocketManager manages WebSocket connections and Kafka consumer
type WebSocketManager struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan []byte
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mu         sync.RWMutex
	consumer   sarama.Consumer
}

// NewWebSocketManager creates a new WebSocket manager
func NewWebSocketManager(kafkaBrokers []string, kafkaTopic string, apiKey, apiSecret string) (*WebSocketManager, error) {
	config := sarama.NewConfig()
	config.Net.SASL.Enable = true
	config.Net.SASL.Mechanism = sarama.SASLTypePlaintext
	config.Net.SASL.User = apiKey
	config.Net.SASL.Password = apiSecret
	config.Net.TLS.Enable = true

	consumer, err := sarama.NewConsumer(kafkaBrokers, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Kafka consumer: %v", err)
	}

	manager := &WebSocketManager{
		clients:    make(map[*websocket.Conn]bool),
		broadcast:  make(chan []byte),
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
		consumer:   consumer,
	}

	// Start consuming from Kafka
	go manager.consumeKafkaMessages(kafkaTopic)
	// Start broadcasting messages
	go manager.broadcastMessages()

	return manager, nil
}

// HandleWebSocket handles WebSocket connections
func (m *WebSocketManager) HandleWebSocket(c *websocket.Conn) {
	// Register new client
	m.register <- c

	// Handle client disconnection
	defer func() {
		m.unregister <- c
		c.Close()
	}()

	// Keep connection alive
	for {
		_, _, err := c.ReadMessage()
		if err != nil {
			break
		}
	}
}

// consumeKafkaMessages consumes messages from Kafka and sends them to broadcast channel
func (m *WebSocketManager) consumeKafkaMessages(topic string) {
	partitions, err := m.consumer.Partitions(topic)
	if err != nil {
		log.Printf("Failed to get partitions for topic %s: %v", topic, err)
		return
	}

	var wg sync.WaitGroup
	for _, partition := range partitions {
		wg.Add(1)
		go func(partition int32) {
			defer wg.Done()
			partitionConsumer, err := m.consumer.ConsumePartition(topic, partition, sarama.OffsetNewest)
			if err != nil {
				log.Printf("Failed to start consumer for topic %s partition %d: %v", topic, partition, err)
				return
			}
			defer partitionConsumer.Close()
			for msg := range partitionConsumer.Messages() {
				var data WaterQualityData
				if err := json.Unmarshal(msg.Value, &data); err != nil {
					log.Printf("Error unmarshaling message: %v", err)
					continue
				}
				if data.ObservationType != "realtime-monitoring" {
					continue
				}
				hasValidFeature := false
				for _, feature := range data.Features {
					if feature.Name == "pH" || feature.Name == "DO" || feature.Name == "EC" {
						hasValidFeature = true
						break
					}
				}
				if !hasValidFeature {
					continue
				}
				m.broadcast <- msg.Value
			}
		}(partition)
	}
	wg.Wait()
}

// broadcastMessages broadcasts messages to all connected clients
func (m *WebSocketManager) broadcastMessages() {
	for {
		select {
		case client := <-m.register:
			m.mu.Lock()
			m.clients[client] = true
			m.mu.Unlock()

		case client := <-m.unregister:
			m.mu.Lock()
			delete(m.clients, client)
			m.mu.Unlock()

		case message := <-m.broadcast:
			m.mu.RLock()
			for client := range m.clients {
				if err := client.WriteMessage(websocket.TextMessage, message); err != nil {
					client.Close()
					delete(m.clients, client)
				}
			}
			m.mu.RUnlock()
		}
	}
}

// Close closes the WebSocket manager and Kafka consumer
func (m *WebSocketManager) Close() error {
	return m.consumer.Close()
}
