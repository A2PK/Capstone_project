FROM golang:1.24 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# This path should match your project structure
RUN CGO_ENABLED=0 GOOS=linux go build -o water-quality-service ./services/water-quality-service/cmd/

FROM alpine:latest

WORKDIR /root/

COPY --from=builder /app/water-quality-service .
COPY --from=builder /app/services/water-quality-service/.env .

ENV $(cat .env | xargs)

CMD ["./water-quality-service"]