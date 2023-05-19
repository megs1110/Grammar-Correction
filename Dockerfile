#FROM python:3.6
FROM huggingface/transformers-pytorch-cpu:latest
COPY ./requirements.txt .
COPY ./models/gpt2-grammar-correction ./models/gpt2-grammar-correction
COPY ./models/gpt2-grammar-correction-tokenizer ./models/gpt2-grammar-correction-tokenizer
ADD ./train.py /
COPY ./templates/index.html ./templates/index.html
COPY ./app.py .
RUN pip install -r ./requirements.txt
#CMD ["python3","./train.py"]

#Expose port outside container
EXPOSE 5000

# Run the app
ENTRYPOINT ["python3", "./app.py"]
