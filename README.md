# Document-Language-Recognition-Methods
Recognition of different document languages based on simple classifiers, The example gives language recognition for 20 different countries.

## 1、Dataset from hugging face for language recognition

For each instance, there is a string for text and a string for tags (language markers). {'labels': 'fr', 'text': 'Conforme à la description, produit pratique.’}
### The Language Identification dataset contains text in 20 languages, which are:
| Language | Abridgement | Language |  Abridgement |
|-------|:---:|-----------|-------:|
| arabic | ar | japanese | ja |
| bulgarian | bg | dutch | nl | 
| german | de | polish | pl | 
| modern greek | el | portuguese | pt | 
| english | en | russian | ru | 
| spanish | es | swahili | sw | 
| french | fr | thai | th | 
| hindi | hi | turkish | tr | 
| italian | it | vietnamese | vi | 
| japanese | ja | chinese | zh | 

### Training set data distribution:
![train_data_distribution](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/8bb17826-87ba-4abc-80ca-06485c5daad8)
![train_num_words](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/2f56ec93-858d-484c-909b-85e70cc0278d)

### Validation set data distribution:
![valid_data_distribution](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/0068a028-3edc-4418-b70b-90e6a1ccfa33)
![valid_num_words](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/a454cfc1-38ef-4b8c-9779-214b517dcf89)

### Test Set Data Distribution:
![test_data_distribution](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/144fcbdb-4c30-4eb8-813b-0f38ccb46863)
![test_num_words](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/95e04aed-5fa1-452e-8037-4e0293362a59)

## 2、Model description.

###  Hyperparameter settings:
max_length = 128,
lr = 5e-5,
num_epochs = 2,
batch_size = 64
Calling a pre-trained tokenizer from hugging face:bert-base-multilingual-cased

### data processing:
Coding the text within the dataset and obtaining a tag index:
{'ar': 0, 'bg': 1, 'de': 2, 'el': 3, 'en': 4, 'es': 5, 'fr': 6, 'hi': 7, 'it': 8, 'ja': 9, 'nl': 10, 'pl': 11, 'pt': 12, 'ru': 13, 'sw': 14, 'th': 15, 'tr': 16, 'ur': 17, 'vi': 18, 'zh': 19}

Most of the dataset text lengths used for language classification are concentrated between 100-150, so the truncated text length of the lexicon is set to 128, and sequences with more than 128 tokens are temporarily ignored to fill in the gap, and any shortfalls are made up to 128 with zeros. attention_mask() is used to make sure that the model's attention is focused on the actual textual content.

### model training:
The model was trained on GPUs, training the model on a bert-base-multilingual-cased basis and optimising the model parameters using a back-propagation algorithm

## 3、Training model performance:

![Loss Graph](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/0df1ca7f-1eea-406e-a8eb-dfdba17a352f)
![Accuracy Graph](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/490dab45-17ce-4c15-9407-ced158a83ed7)

Classification Report:
![WX20230825-154558@2x](https://github.com/sy9372/Document-Language-Recognition-Methods/assets/95511690/4a068b38-2ea6-4f02-9805-49e2acf4821f)
