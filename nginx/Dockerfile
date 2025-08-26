FROM nginx:alpine

# Copy nginx configuration
COPY nginx/nginx-proxy.conf /etc/nginx/nginx.conf

# Copy certificates
# COPY cert-generator/certs /etc/nginx/certs
# Certificates will be mounted as volume at runtime
RUN mkdir -p /etc/nginx/certs

# Expose port
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
