# Stage 1: Build
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files first for better layer caching
COPY package*.json ./

# Use npm ci instead of npm install (faster and more reliable)
RUN npm ci --prefer-offline --no-audit

# Copy source code after dependencies
COPY . .

# Build argument
ARG VITE_BACKEND_BASE_API
ENV VITE_BACKEND_BASE_API=$VITE_BACKEND_BASE_API

# Build the app
RUN npm run build

# Stage 2: Nginx
FROM nginx:alpine

# Copy build output to Nginx html directory
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]