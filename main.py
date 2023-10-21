import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Empty the GPU
torch.cuda.empty_cache()
# Reading training and validation files
train_data_path = "./train.csv"
valid_data_path = "./valid.csv"
test_data_path = "./test.csv"
train_data = pd.read_csv(train_data_path)[:5000]
valid_data = pd.read_csv(valid_data_path)[:2000]
test_data = pd.read_csv(test_data_path)[:2000]

# Define the dataset class
class LanguageRecognitionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=100):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = sorted(list(set(self.data['labels']))) # Get all the different tags

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.loc[index, 'text']
        label = self.data.loc[index, 'labels']

        # Encoding of text
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Get tag index
        label_idx = self.labels.index(label)
        # Focuses the model's attention on the actual textual content
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_idx, dtype=torch.long)
        }

# Creating a Data Loader
def create_data_loader(data, tokenizer, batch_size=16):
    dataset = LanguageRecognitionDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size)

# Set up a tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize all sub-datasets:
train_data_loader = create_data_loader(train_data, tokenizer)
valid_data_loader = create_data_loader(valid_data, tokenizer)
test_data_loader = create_data_loader(test_data,tokenizer)





# Preparing the model
num_labels = len(set(train_data['labels']))
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu') 
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Data storage
train = []
train_true_labels = []
train_loss_ls = []
train_accuracy_ls = []

valid = []
valid_true_labels = []
valid_loss_ls = []
valid_accuracy_ls = []

predictions = []
true_labels = []
test_loss_ls = []
test_accuracy_ls = []

# Training Models
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')
        # forward propagation
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training labels
        train_loss = loss.item()  
        train_loss_ls.extend([train_loss])
        _, train_labels = torch.max(outputs.logits, dim=1)
        train.extend(train_labels.cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())
        train_accuracy = accuracy_score(train_true_labels, train).item()
        train_accuracy_ls.extend([train_accuracy])

    # Model Validation 
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in valid_data_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
             # Validation labels
            valid_loss = outputs.loss.item()  
            valid_loss_ls.extend([valid_loss])
            _, valid_labels = torch.max(outputs.logits, dim=1)
            valid.extend(valid_labels.cpu().numpy())
            valid_true_labels.extend(labels.cpu().numpy())
            valid_accuracy = accuracy_score(valid_true_labels, valid).item()
            valid_accuracy_ls.extend([valid_accuracy])

    # Model Testing
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            test_loss = outputs.loss.item()
            total_test_loss += outputs.loss.item()
            
            # Testing labels
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            test_loss_ls.extend([test_loss])
            test_accuracy = accuracy_score(true_labels, predictions).item()
            test_accuracy_ls.extend([test_accuracy])

# Modelling performance
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average='macro')
precision = precision_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')
print(f'accuracy: {accuracy:.4f}')
print(f'recall rate: {recall:.4f}')
print(f'precision rate: {precision:.4f}')
print(f'F1-score: {f1:.4f}')

# Print report
label_mapping = {label: idx for idx, label in enumerate(test_data['labels'].unique())}
label_names = [name for name, index in sorted(label_mapping.items(), key=lambda x: x[1])]
report = classification_report(true_labels, predictions, target_names=label_names)
print(report)
  
print(f'Training Losses: {train_loss_ls}')
print(f'Training accuracy: {train_accuracy_ls}')
print(f'Verification losses: {valid_loss_ls}')
print(f'Verification of accuracy: {valid_accuracy_ls}')
print(f'Verification losses: {test_loss_ls}')
print(f'Verification of accuracy: {test_accuracy_ls}')

# Loss Graph
plt.plot(train_loss_ls,'r',label='train loss')
plt.plot(test_loss_ls,'b',label='test loss')
plt.xlabel('No. of batchs')
plt.ylabel('Categorical Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

# Accuracy Graph
plt.plot(train_accuracy_ls,'r',label='train loss')
plt.plot(test_accuracy_ls,'b',label='test loss')
plt.xlabel('No. of batchs')
plt.ylabel('Categorical Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.show()

