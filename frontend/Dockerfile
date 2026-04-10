# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (cacheable layer)
COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps

# Copy source and build
COPY . .
RUN npm run build

# Runtime stage — serve with nginx
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config for SPA routing and API proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
