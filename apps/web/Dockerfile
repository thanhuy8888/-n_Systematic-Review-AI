# ─── Stage 1: Build React app ─────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build
# Output: /app/dist/

# ─── Stage 2: Serve with Nginx ────────────────────────────────────────────
FROM nginx:1.25-alpine AS production

# Copy built React files
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config: SPA routing + API proxy to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
