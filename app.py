import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def correct_grammar(txt):
  ids = tokenizer.encode('<|startoftext|> '+ txt + ' Corrected:', return_tensors='pt')
  if txt[-1]!='.':
      txt+="."
  with torch.no_grad(): 
  	sample_outputs = model.generate( ids,
                                        pad_token_id=tokenizer.eos_token_id,
                                        do_sample=True,
                                        max_length = 50,
                                        top_p=0.95,
                                        num_return_sequences=1
                                    )
  	for i, sample_output in enumerate(sample_outputs):
              return tokenizer.decode(sample_output, skip_special_tokens=True)
              print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_txt = list(request.form.values())[0]
    pred = correct_grammar(inp_txt).split("Corrected:")[1]
    return render_template('index.html', prediction_text='Corrected sentence is \"{}\"'.format(pred))


if __name__ == "__main__":
    global model, tokenizer
   
    model = GPT2LMHeadModel.from_pretrained('./models/gpt2-grammar-correction/')
    tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2-grammar-correction-tokenizer/')
    model.eval()
    app.run(debug=False, host='0.0.0.0')
