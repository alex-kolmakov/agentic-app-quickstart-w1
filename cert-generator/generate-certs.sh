#!/bin/sh

# Create CA private key
openssl genrsa -out ca.key 4096

# Create CA certificate
openssl req -new -x509 -days 365 -key ca.key -sha256 -out ca.crt -subj "/C=US/ST=CA/L=SF/O=MyOrg/CN=MyCA"

# Create server private key
openssl genrsa -out nginx-proxy.key 4096

# Create server certificate signing request
openssl req -subj "/C=US/ST=CA/L=SF/O=MyOrg/CN=nginx-proxy" -new -key nginx-proxy.key -out nginx-proxy.csr

# Create server certificate
openssl x509 -req -days 365 -in nginx-proxy.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out nginx-proxy.crt -sha256

# Create client private key
openssl genrsa -out agentic-app.key 4096

# Create client certificate signing request
openssl req -subj "/C=US/ST=CA/L=SF/O=MyOrg/CN=agentic-app" -new -key agentic-app.key -out agentic-app.csr

# Create client certificate
openssl x509 -req -days 365 -in agentic-app.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out agentic-app.crt -sha256

echo "Certificates generated successfully!"
ls -la /certs/
