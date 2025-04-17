FROM golang:1.24 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# This path should match your project structure
# Changed the output binary name and the build path
RUN CGO_ENABLED=0 GOOS=linux go build -o model-service ./services/model-service/cmd/

FROM alpine:latest

WORKDIR /root/

# Changed the binary name to copy
COPY --from=builder /app/model-service .
# Changed the env file path to copy
COPY --from=builder /app/services/model-service/.env .

ENV $(cat .env | xargs)

# Changed the command to run the correct binary
CMD ["./model-service"] 