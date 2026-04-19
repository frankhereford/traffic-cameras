#!/bin/sh
set -e

DOMAIN="signals.frankhereford.com"
EMAIL="frank.hereford@austintexas.gov"
PEM_FILE="/certs/${DOMAIN}.pem"
CERT_DIR="/etc/letsencrypt/live/${DOMAIN}"

# -checkend returns 0 if the cert is valid for at least that many more seconds
if [ -f "$PEM_FILE" ] && openssl x509 -checkend 2592000 -noout -in "$PEM_FILE" 2>/dev/null; then
    echo "Certificate valid for more than 30 days — skipping."
    exit 0
fi

certbot certonly \
    --dns-route53 \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    -d "$DOMAIN" \
    -d "*.${DOMAIN}"

# HAProxy expects cert + key concatenated in a single PEM
cat "${CERT_DIR}/fullchain.pem" "${CERT_DIR}/privkey.pem" > "$PEM_FILE"
echo "Certificate written to $PEM_FILE"
