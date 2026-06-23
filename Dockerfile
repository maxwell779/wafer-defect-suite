# wafer-defect-suite — 프론트(빌드) + FastAPI 백엔드 단일 컨테이너
# 빌드: docker build -t wafer-suite .
# 실행: docker run -p 8000:8000 wafer-suite   → http://localhost:8000 (DEMO)
#   LIVE 추론은 가중치 필요: -v %cd%/experiments:/app/experiments -v %cd%/data:/app/data

# 1) 프론트엔드 빌드
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2) 백엔드 + 정적 프론트
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# CPU torch(가벼운 데모) — GPU 추론은 호스트에서 직접 실행 권장
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart \
    numpy pandas scikit-learn pillow \
    torch --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart numpy pandas scikit-learn pillow torch
COPY backend/ ./backend/
COPY src/ ./src/
COPY config.py ./
COPY web/src/appdata ./web/src/appdata
COPY --from=web /web/dist ./web/dist
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
