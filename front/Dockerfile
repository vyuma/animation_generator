# ============================================
# Frontend (Next.js) Dockerfile
# ============================================
FROM node:22-bookworm-slim

WORKDIR /app

# 依存関係は bun.lock で管理
RUN npm install -g bun@1
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# API Routes からバックエンドへのプロキシ先（サーバーサイドのみで使用）
ENV API_URL=http://backend:8888 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
EXPOSE 3000

CMD ["npm", "run", "start"]
