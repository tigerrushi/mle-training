FROM python:latest

LABEL Maintainer = "rushikesh"

WORKDIR /usr/app

COPY myproject.py ./
COPY mletraining-0.1.12-py3-none-any.whl ./
RUN mkdir processed saved_models
RUN pip install ./mletraining-0.1.12-py3-none-any.whl

CMD ["python", "./myproject.py"]