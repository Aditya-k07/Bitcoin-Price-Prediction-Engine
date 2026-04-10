# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

RUN apk add --no-cache git

# Download dependencies first (cacheable layer)
COPY go.mod go.sum* ./
RUN go mod download

# Build the binary with optimizations
COPY cmd/ ./cmd/
COPY internal/ ./internal/
COPY config/ ./config/
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags='-s -w' -o /coinsight-backend ./cmd/server

# Runtime stage (minimal image)
FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /coinsight-backend /coinsight-backend

EXPOSE 8080

ENTRYPOINT ["/coinsight-backend"]
