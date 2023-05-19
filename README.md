# Grammar Correction Model

This repository contains a grammar correction model that has been fine-tuned on the [Owishiboo/grammar-correction](https://huggingface.co/datasets/Owishiboo/grammar-correction) dataset using a pretrained GPT-2 model.
Since the dataset is small with nearly ~6000 datapoints the performance of the model is bottlenecked by it. The model has been trained on colab with gpu for 4 epochs. 

## Model Training

To train the model, follow these steps:

1. Download the model and tokenizer files from the following Google Drive links:
   - [Saved Model Files](https://drive.google.com/drive/folders/1SwEmxDDY7VbTJeSkHS3Bs_r5QSUIp76J?usp=share_link)
   - [Tokenizer Files](https://drive.google.com/drive/folders/18fFDQDoIlwwxJLgm1NCoolo1aqbZNDZt?usp=share_link)
4. Copy the downloaded folders to the root directory of the "grammar-correction" directory, inside the "/models/" subdirectory.

To run the training script, make sure you have installed the required dependencies:

1. pip install -r requirements.txt

Then, execute the training script: **"python train.py --num_epochs=1"**


Note: The `train.py` file has been edited to use CPU for testing on local systems without NVIDIA GPUs.

## Interacting with the Model

To interact with the fine-tuned model, you can use a Docker container. Follow these steps:

1. Build the Docker image: **"docker build -t grammar ."** 
2. Run the Docker container: **"docker run -it -p 80:5000 grammar"**


3. Open your browser and visit [localhost:80](http://localhost:80) to connect to the Flask server hosting the fine-tuned model.

## Example Outputs

Here are a few examples where the model performs well:

**Input:** I are going to school.
**Output:** I am going to school.

**Input:** I done this to help my family.
**Output:** I did this to help my family.

Note: Please use generic sentences.

## Further Enhancements

To enhance the training and evaluation of the model, consider the following steps:

1. Use a more comprehensive dataset like [C4_200M](https://ai.googleblog.com/2021/08/the-c4200m-synthetic-dataset-for.html).
2. Train the model for more epochs, utilize early stopping, and monitor training and validation set loss to fine-tune the hyperparameters.
3. Use BLEU scores as an evaluation metric for the trained model.

Feel free to explore these enhancements to improve the performance of the grammar correction model.

