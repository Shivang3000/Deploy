# Stage 1: Build
FROM node:18 AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

# Build arguments for environment variables
ARG VITE_BACKEND_BASE_API

# This line passes an environment variable into the Docker container so the React app knows the backend API URL.
ENV VITE_BACKEND_BASE_API=$VITE_SERVER_BASE_URL

COPY . .

RUN npm run build

# Stage 2: Nginx, alpine means the lighter version of Nginx
FROM nginx:alpine

# Copy build output to Nginx html directory
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]