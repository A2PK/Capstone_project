# Water Quality & AI Forecasting Microservice Project

This project demonstrates a microservices architecture for water quality monitoring, forecasting, and user management, deployed on Kubernetes. It uses Go for backend services, Python for AI/ML, and PostgreSQL for the database. The system is designed for local development using kind (Kubernetes in Docker) and supports real-time data streaming, user management, and AI-powered water quality forecasting.

## Services

The project consists of the following backend services:

*   `api-gateway`: Handles incoming HTTP/JSON requests and routes them to the appropriate backend gRPC service using `gRPC-Gateway` as a reverse proxy. Also provides a WebSocket endpoint for real-time water quality data.
*   `user-service`: Manages user data, authentication, and user-related actions (registration, login, profile, etc).
*   `water-quality-service`: Handles ingestion, storage, and querying of water quality data (pH, DO, EC, etc) from various stations. Integrates with Kafka for real-time data.
*   `ai-service`: Provides AI/ML model training and forecasting for water quality. Supports scheduled retraining and prediction for all stations, and exposes APIs for on-demand model management and forecasting.
*   `database`: PostgreSQL database instance managed via Kubernetes manifests.

## Key Features

- **User Management:** Register, authenticate, and manage users via the `user-service`.
- **Water Quality Insights:**
  - Real-time ingestion of water quality data (pH, DO, EC, etc) from sensors or synthetic producers via Kafka.
  - Query historical and current water quality data for any station.
  - WebSocket endpoint (`/ws/water-quality`) for real-time updates to dashboards or clients.
- **AI Forecasting Models:**
  - Scheduled daily retraining and prediction for all stations using the latest data (via APScheduler in `ai-service`).
  - On-demand model training, evaluation, and prediction via REST/gRPC APIs.
  - Model metrics and recommendations for best-performing models per parameter/station.
- **Real-Time Data Flow:**
  - Data is produced to Kafka (e.g., by a Python producer) and consumed by the backend for storage and real-time broadcast.
  - WebSocket clients receive new data as soon as it arrives.

## Local Development Setup

This setup uses [kind](https://kind.sigs.k8s.io/) to run a local Kubernetes cluster using Docker containers as nodes.

### Requirements

1.  **Docker:** Install Docker Desktop or Docker Engine. `kind` runs Kubernetes within Docker containers. Follow the official Docker installation guide: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
2.  **kind:** Install `kind` to create and manage local Kubernetes clusters. Follow the official installation guide: [https://kind.sigs.k8s.io/docs/user/quick-start/#installation](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
3.  **kubectl:** Install the Kubernetes command-line tool. Follow the official guide: [https://kubernetes.io/docs/tasks/tools/install-kubectl/](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
4.  **Go:** Ensure you have Go installed (check `go.mod` files for required version).
5.  **Python 3.12+** (for AI/ML service and producer scripts).
6.  **Make:** Used for running installation and build commands.

### Dependencies Installation

Before running the project, install the necessary Go and Python dependencies:

```bash
make install-deps
# For Python AI service:
cd services/ai-service && poetry install
```

This command installs:
*   Buf CLI (for Protocol Buffer management)
*   gRPC-Gateway dependencies
*   Swagger UI dependencies
*   Other Go module dependencies
*   Python dependencies for AI/ML and Kafka producer

### Configuration (`.env.example`)

Each service directory (e.g., `api-gateway/`, `user-service/`, `ai-service/`, etc.) requires environment variables for configuration (database connection strings, secrets, Kafka, etc).

*   Look for a `.env.example` file within each service's directory.
*   Copy this file to `.env` (`cp .env.example .env`) in the same directory.
*   Modify the `.env` file with your local configuration values (especially secrets and Kafka credentials).

### Running the Project Locally with kind

1.  **Start kind Cluster:**
    Create a local Kubernetes cluster using `kind`:
    ```bash
    kind create cluster --name ride-sharing
    ```
    This also sets your `kubectl` context to `kind-ride-sharing`.

2.  **(Optional) Build and Load Service Images:**
    If you modify the Go or Python services and want to run them within the kind cluster, you need to build their Docker images and load them into the kind cluster. (Alternatively, configure your Kubernetes deployments to pull from a registry where you push your images).
    You might need a `Makefile` target or script to build images for all services (e.g., `docker build -t ai-service:latest ./ai-service`).
    Then, load each image:
    ```bash
    kind load docker-image ai-service:latest --name ride-sharing
    kind load docker-image api-gateway:latest --name ride-sharing
    # ... repeat for all services
    ```

3.  **Deploy to Kubernetes:**
    Apply the Kubernetes manifests to your `kind` cluster. Ensure you apply them in a logical order (e.g., namespace, secrets, configmaps, database, then services).
    ```bash
    kubectl apply -f k8s/database/
    kubectl apply -f k8s/api-gateway/
    kubectl apply -f k8s/user-service/
    kubectl apply -f k8s/ai-service/
    kubectl apply -f k8s/water-quality-service/
    # ...
    ```
    *Wait for pods to become ready:* `kubectl get pods -n ride-sharing -w`

### Accessing Services

*   **API Gateway:** The primary entry point for HTTP requests. Forward a local port to the `api-gateway-service` running in the cluster:
    ```bash
    kubectl port-forward service/api-gateway-service -n ride-sharing 8080:<service_port>
    ```
    Access the API at `http://localhost:8081`.
*   **WebSocket Real-Time Data:** Connect to `ws://localhost:8081/ws/water-quality` to receive real-time water quality updates.
*   **AI/ML Service:**
    - REST/gRPC endpoints for model training, prediction, and metrics.
    - Scheduled jobs for daily retraining and prediction (see logs for status).
*   **Database:** PostgreSQL runs within the cluster. For direct local access:
    ```bash
    kubectl port-forward service/postgres-service -n ride-sharing 5432:5432
    ```
*   **Other Services:** Access via API Gateway or directly within the cluster using their service names.

### Real-Time Data Flow Example

1. **Start the Python producer** (in `backend/`):
    ```bash
    poetry run python producer.py
    ```
    This will generate and send synthetic water quality data to Kafka for all stations.
2. **Connect a WebSocket client** to `/ws/water-quality` to see real-time updates.
3. **AI/ML service** will retrain and predict daily at midnight (Asia/Ho_Chi_Minh timezone) and can be triggered via API as well.

### Cleaning Up

To delete the local `kind` cluster and all its resources:
```bash
kind delete cluster --name ride-sharing
```