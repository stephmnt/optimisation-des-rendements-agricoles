FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=127.0.0.1 \
    API_PORT=8000 \
    STREAMLIT_HOST=127.0.0.1 \
    STREAMLIT_PORT=8502 \
    PUBLIC_PORT=8501 \
    API_BASE_URL=http://127.0.0.1:8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY nginx.conf /etc/nginx/nginx.conf
COPY main.py ./main.py
COPY streamlit ./streamlit
COPY data ./data
COPY artifacts ./artifacts
COPY agriculture.png ./agriculture.png

EXPOSE 8000 8501

HEALTHCHECK CMD curl --fail http://127.0.0.1:8000/health && curl --fail http://127.0.0.1:8502/_stcore/health && curl --fail http://127.0.0.1:8501/api/health || exit 1

CMD ["sh", "-lc", "uvicorn main:app --host \"$API_HOST\" --port \"$API_PORT\" & api_pid=$!; streamlit run streamlit/src/streamlit_app.py --server.address=\"$STREAMLIT_HOST\" --server.port=\"$STREAMLIT_PORT\" --server.headless=true & streamlit_pid=$!; trap 'kill \"$api_pid\" \"$streamlit_pid\" 2>/dev/null || true' EXIT INT TERM; attempt=0; while [ \"$attempt\" -lt 30 ]; do if curl --silent --fail \"$API_BASE_URL/health\" >/dev/null; then break; fi; attempt=$((attempt + 1)); sleep 1; done; if [ \"$attempt\" -eq 30 ]; then echo \"FastAPI ne répond pas sur $API_BASE_URL/health\" >&2; exit 1; fi; attempt=0; while [ \"$attempt\" -lt 30 ]; do if curl --silent --fail \"http://127.0.0.1:$STREAMLIT_PORT/_stcore/health\" >/dev/null; then break; fi; attempt=$((attempt + 1)); sleep 1; done; if [ \"$attempt\" -eq 30 ]; then echo \"Streamlit ne répond pas sur http://127.0.0.1:$STREAMLIT_PORT/_stcore/health\" >&2; exit 1; fi; exec nginx -g 'daemon off;'"]
