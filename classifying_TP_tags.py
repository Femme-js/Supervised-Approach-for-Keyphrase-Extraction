import numpy as np
import pandas as pd
import torch
import argparse
import ast
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
import torch.nn.functional as F
from rich.console import Console
from transformers import AutoModel, BertTokenizerFast, AutoTokenizer
from itertools import zip_longest
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

console = Console()



# specify GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')



class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      # apply softmax activation
      x = self.softmax(x)
      
  

      return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)



def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def pre_processing_data(data):

  
  data['text_input'] = 'nan'
  for i in data.index:
      l1 = data['text_with_ws'][i]
      l2 = data['lemma_'][i]
      l3 = data['pos_'][i]
      l4 = data['tag_'][i]
      l5 = data['dep_'][i]
      l6 = data['shape_'][i]
      l7 = data['ent_type_'][i]
  

      lists = [l1,l2,l3,l4,l5,l6,l7]
      
      data['text_input'][i] = [ str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) + " " + str(g) for  b, c, d, e, f ,g in zip( l2, l3, l4, l5,l6,l7)]

      data['text_input'][i] = ", ".join(data['text_input'][i])

      
  data['label'] = 1

  return data

	#features used as classifier input


def data_preparation(data):

	test_text = data['text_input']
	test_labels = data['label']

	# tokenize and encode sequences in the test dataset
	tokens_test = tokenizer.batch_encode_plus(
    	test_text.tolist(),
    	max_length = 15,
    	pad_to_max_length=True,
    	truncation=True
	)

	test_seq = torch.tensor(tokens_test['input_ids'])
	test_mask = torch.tensor(tokens_test['attention_mask'])
	test_y = torch.tensor(test_labels.tolist())

	#define a batch size
	batch_size = 32

	# wrap tensors
	test_data = TensorDataset(test_seq, test_mask, test_y)

	# sampler for sampling the data during testing
	test_sampler = RandomSampler(test_data)

	# dataLoader for test set
	test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

	return test_dataloader, test_text




def tags_score(model, test_dataloader, test_text):
  

	# Compute predicted probabilities on the test set
	probs = bert_predict(model, test_dataloader)

	# Get predictions from the probabilities
	threshold = 0.5
	preds = np.where(probs[:, 1] > threshold, 1, 0)

	d = {'source' : test_text.tolist(), 'score':  probs[:, 1]}

	dfx = pd.DataFrame(d)

	return dfx






if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # currently, we need path to dataset, flag to use a custom model, path to custom model, and flag to compute bias metrics
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--model_path", type=str)

    # read the arguments from the command line
    args = parser.parse_args()

    
    file_path=args.file_path
    data = pd.read_csv(file_path)

    data = pre_processing_data(data)
    model_path=args.model_path
    torch.save(model.state_dict(), model_path)
    
    test_dataloader, test_text = data_preparation(data)
    dfx = tags_score(model, test_dataloader, test_text)
    dfa = pd.merge(data, dfx, on='source', how='left' )
    dfb = pd.merge(data, dfx, on='source', how='right' )
    dfa['score'] = dfb['score']
    dfa.dropna()
    df4 = dfa

    print(df4.columns)

    dfinal = pd.DataFrame()

    dfinal['source'] = df4['source']
    dfinal['Tags'] = df4['Tags']
    dfinal['score'] = df4['score']

    dfinal = dfinal.dropna()

    dfinal['dict_score'] = '"' + df4['Tags'].astype(str) + '"' + ": " +  df4['score'].astype(str)

    dfinal['Tags'] = dfinal.groupby(['source'])['Tags'].transform(lambda x : ', '.join(x.astype(str)))
    dfinal['score'] = dfinal.groupby(['source'])['score'].transform(lambda x : ', '.join(x.astype(str)))
    dfinal['dict_score'] = dfinal.groupby(['source'])['dict_score'].transform(lambda x : ', '.join(x.astype(str)))

    dfinal = dfinal.drop_duplicates()


    dfinal['dict_score'] = '{' + dfinal['dict_score'].astype(str) + '}'
    dfinal['dict_score'] = dfinal['dict_score'].apply(ast.literal_eval)

    dfinal['sort_score'] = 'nan'
    dfinal['topN_tags'] = 'nan'
    for i in dfinal.index:
        dfinal['sort_score'][i] = sorted(dfinal['dict_score'][i].items(), key=lambda x: x[1], reverse=True)
        dfinal['topN_tags'][i] = []
        for value in dfinal['sort_score'][i][:20]:
            dfinal['topN_tags'][i].append(value[0])


    dfinal.to_csv("semeval_topN_tags.csv")







    


        






  




