# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install git for module downloads
RUN apk add --no-cache git

# Download dependencies first (cacheable layer)
COPY go.mod ./
COPY go.sum* ./
RUN go mod download

# Build the binary
COPY . .
RUN go mod tidy && CGO_ENABLED=0 GOOS=linux go build -o /coinsight-backend ./cmd/server

# Runtime stage (minimal image)
FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /coinsight-backend /coinsight-backend

EXPOSE 8080

ENTRYPOINT ["/coinsight-backend"]
