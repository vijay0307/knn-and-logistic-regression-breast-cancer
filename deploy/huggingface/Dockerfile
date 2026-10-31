# Hugging Face Spaces (Docker SDK) image for the RAG chatbot + dashboard.
# Spaces run the container as uid 1000 and route public traffic to app_port (7860).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements-rag.txt .
RUN pip install --no-cache-dir -r requirements-rag.txt
COPY --chown=user rag ./rag
COPY --chown=user PROJECT_README.md ./PROJECT_README.md
USER user
# /tmp is writable on Spaces. Usage and feedback logs reset when the Space restarts;
# enable Persistent Storage and point these at /data to keep them.
ENV RAG_ENV=production \
    RAG_SOURCES=/home/user/app/rag/knowledge,/home/user/app/PROJECT_README.md \
    RAG_EVENTS_DB=/tmp/data/rag_events.db \
    RAG_FEEDBACK_LOG=/tmp/data/rag_feedback.jsonl \
    RAG_REPORT_DIR=/home/user/app/reports
COPY --chown=user reports ./reports
EXPOSE 7860
CMD ["uvicorn", "rag.server:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
