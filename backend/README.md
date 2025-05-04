# Ride Sharing Sample Kubernetes Microservice Project

This project demonstrates a microservices architecture deployed on Kubernetes, using Go for backend services and PostgreSQL for the database. It's designed for local development using kind (Kubernetes in Docker).

## Services

The project consists of the following backend services:

*   `api-gateway`: Handles incoming HTTP/JSON requests and routes them to the appropriate backend gRPC service using `gRPC-Gateway` as a reverse proxy.
*   `user-service`: Manages user data and authentication.
*   `patient-service`: Manages patient-related data.
*   `staff-service`: Manages staff-related data.
*   `appointment-service`: Manages appointment scheduling.
*   `database`: PostgreSQL database instance managed via Kubernetes manifests.

## Local Development Setup

This setup uses [kind](https://kind.sigs.k8s.io/) to run a local Kubernetes cluster using Docker containers as nodes.

### Requirements

1.  **Docker:** Install Docker Desktop or Docker Engine. `kind` runs Kubernetes within Docker containers. Follow the official Docker installation guide: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
2.  **kind:** Install `kind` to create and manage local Kubernetes clusters. Follow the official installation guide: [https://kind.sigs.k8s.io/docs/user/quick-start/#installation](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
3.  **kubectl:** Install the Kubernetes command-line tool. Follow the official guide: [https://kubernetes.io/docs/tasks/tools/install-kubectl/](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
4.  **Go:** Ensure you have Go installed (check `go.mod` files for required version).
5.  **Make:** Used for running installation and build commands.

### Dependencies Installation

Before running the project, install the necessary Go and Protobuf dependencies:

```bash
make install-deps
```

This command installs:
*   Buf CLI (for Protocol Buffer management)
*   gRPC-Gateway dependencies
*   Swagger UI dependencies
*   Other Go module dependencies

### Configuration (`.env.example`)

Each service directory (e.g., `api-gateway/`, `user-service/`, etc.) requires environment variables for configuration (database connection strings, secrets, etc.).

*   Look for a `.env.example` file within each service's directory.
*   Copy this file to `.env` (`cp .env.example .env`) in the same directory.
*   Modify the `.env` file with your local configuration values (especially secrets and potentially database hostnames if not using Kubernetes service discovery).

**Note:** The Kubernetes manifests often pull sensitive configuration (like DB passwords) from Kubernetes Secrets, which are defined in the `/k8s` directory. Ensure consistency between your `.env` files (if used for local `go run`) and the Kubernetes secrets.

### Running the Project Locally with kind

1.  **Start kind Cluster:**
    Create a local Kubernetes cluster using `kind`:
    ```bash
    kind create cluster --name ride-sharing-dev
    ```
    This also sets your `kubectl` context to `kind-ride-sharing-dev`.

2.  **(Optional) Build and Load Service Images:**
    *If you modify the Go services and want to run them within the kind cluster*, you need to build their Docker images and load them into the kind cluster. (Alternatively, configure your Kubernetes deployments to pull from a registry where you push your images).
    You might need a `Makefile` target or script to build images for all services (e.g., `docker build -t user-service:latest ./user-service`).
    Then, load each image:
    ```bash
    kind load docker-image user-service:latest --name ride-sharing-dev
    kind load docker-image patient-service:latest --name ride-sharing-dev
    # ... repeat for all services including api-gateway
    ```

3.  **Deploy to Kubernetes:**
    Apply the Kubernetes manifests to your `kind` cluster. Ensure you apply them in a logical order (e.g., namespace, secrets, configmaps, database, then services).
    ```bash
    # Assuming a namespace file exists (adjust path if needed)
    # kubectl apply -f k8s/common/namespace.yaml

    # Apply database manifests (Secret, Service, StatefulSet)
    kubectl apply -f k8s/database/

    # Apply manifests for each service (Deployments, Services, etc.)
    kubectl apply -f k8s/api-gateway/
    kubectl apply -f k8s/user-service/
    kubectl apply -f k8s/patient-service/
    kubectl apply -f k8s/staff-service/
    kubectl apply -f k8s/appointment-service/
    ```
    *Wait for pods to become ready:* `kubectl get pods -n ride-sharing -w`

### Accessing Services

*   **API Gateway:** The primary entry point for HTTP requests. You'll likely need to forward a local port to the `api-gateway-service` running in the cluster:
    ```bash
    # Find the service port (e.g., 8080) using 'kubectl get svc -n ride-sharing api-gateway-service'
    kubectl port-forward service/api-gateway-service -n ride-sharing 8080:<service_port>
    ```
    You can then access the API at `http://localhost:8080`.
*   **Database:** The PostgreSQL database runs within the `kind` cluster. Services connect using the Kubernetes service name (`postgres-service.ride-sharing:5432`). For direct local access (e.g., using a GUI tool), set up port forwarding:
    ```bash
    kubectl port-forward service/postgres-service -n ride-sharing 5432:5432
    ```
*   **Other Services:** Backend gRPC services are typically accessed via the API Gateway, but can be accessed directly within the cluster using their service names (e.g., `user-service.ride-sharing:<grpc_port>`).

### Cleaning Up

To delete the local `kind` cluster and all its resources:
```bash
kind delete cluster --name ride-sharing-dev
```