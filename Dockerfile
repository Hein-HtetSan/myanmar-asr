# ─────────────────────────────────────────────────────
#  Myanmar ASR — Presentation + Demo (All-in-One)
#
#  Contains:
#   • Next.js slide deck  → port 3000 (internal)
#   • Streamlit demo app  → port 8501 (internal)
#   • Nginx reverse proxy → port 80   (exposed)
#
#  Routes:
#   /            → Presentation (Next.js)
#   /demo        → Streamlit ASR Demo
# ─────────────────────────────────────────────────────

# ── Stage 1: Build Next.js Presentation ──────────────
FROM node:20-alpine AS presentation-builder

WORKDIR /build
COPY presentation/package.json presentation/package-lock.json ./
RUN npm ci --ignore-scripts

COPY presentation/ .
RUN npm run build

# ── Stage 2: Runtime ─────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx curl nodejs npm supervisor \
    && rm -rf /var/lib/apt/lists/*

# ── Install Node.js 20 (for Next.js production server) ──
RUN npm install -g n && n 20 && hash -r

# ── Python deps for Streamlit app ────────────────────
COPY requirements-demo.txt /tmp/requirements-demo.txt
RUN pip install --no-cache-dir -r /tmp/requirements-demo.txt

# ── Copy Next.js build output ────────────────────────
WORKDIR /app/presentation
COPY --from=presentation-builder /build/.next ./.next
COPY --from=presentation-builder /build/public ./public
COPY --from=presentation-builder /build/package.json ./package.json
COPY --from=presentation-builder /build/package-lock.json ./package-lock.json
COPY --from=presentation-builder /build/next.config.ts ./next.config.ts
COPY --from=presentation-builder /build/node_modules ./node_modules

# ── Copy Streamlit app ───────────────────────────────
WORKDIR /app
COPY scripts/deploy/streamlit_app.py /app/streamlit_app.py
# .vastai_state is optional (only exists locally); use a glob with the dot-docker trick
COPY .vastai_stat[e] /app/

# Create directories the Streamlit app expects
RUN mkdir -p /app/models/mlflow_cache /app/results

# ── Streamlit config (headless, no browser) ──────────
RUN mkdir -p /root/.streamlit && \
    printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml && \
    printf '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

# ── Nginx config ─────────────────────────────────────
COPY docker/nginx.conf /etc/nginx/sites-available/default

# ── Supervisor config (manages all processes) ────────
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Entrypoint ───────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
