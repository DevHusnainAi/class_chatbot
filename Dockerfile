FROM langchain/langgraph-api:3.11

ADD . /deps/class_chatbot

RUN pip install --upgrade pip

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"customer_service_agent": "./src/class_chatbot/main.py:graph"}'

WORKDIR /deps/class_chatbot