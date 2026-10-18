# Dashboard frontend: Vite production build served by nginx, which also proxies /api (HTTP and the
# live WebSocket) to the backend, so the browser only ever talks to this one origin.
#   docker build -t robotics-frontend ./frontend
#   docker run -p 3000:3000 -e BACKEND_ADDR=host.docker.internal:8000 robotics-frontend

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# BACKEND_ADDR is substituted into the template at container start (nginx image's envsubst step).
ENV BACKEND_ADDR=backend:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 3000
