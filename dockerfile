
FROM python:3.9

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN echo "#!/bin/sh\nexit 0" > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo


COPY . /app

EXPOSE 8501

CMD ["streamlit","run","uicopy.py","--server.port=8501", "--server.address=0.0.0.0"]