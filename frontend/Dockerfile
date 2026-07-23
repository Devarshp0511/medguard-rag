# MedGuard RAG frontend — a single static HTML file, served by nginx.
# No build step needed since index.html has no framework/bundler dependency.

FROM nginx:alpine

COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80
