FROM python:3.13-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user shor_app.py shor_bluequbit.py ./
USER user
ENV SHOR_PUBLIC=1 PORT=10000 PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["python", "shor_app.py"]
