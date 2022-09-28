import pandas as pd 
import ast
import argparse
import numpy as np
import spacy
from rich.console import Console
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

console = Console()

nlp = spacy.load('en_core_web_sm')

def print_pos_df(doc):
  recs = []
  for token in doc:
    recs.append([token.text, token.text_with_ws, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop, token.ent_type_, token.ent_iob_])
    cols = ['text', 'text_with_ws', 'lemma_', 'pos_', 'tag_', 'dep_', 'shape_', 'is_alpha', 'is_stop', 'ent_type_', 'ent_iob_']
    pos_df = pd.DataFrame(recs, columns=cols)

  features_dict = pos_df.to_dict(orient = 'index') 
  return features_dict

def generate_features(feature, features_dict, list_x):
    a = []
    b = []

    test = list(features_dict.values())

    for val in list_x:
      for i in val:
        for j in range(len(test)):
          if ps.stem(test[j]['text']) == ps.stem(i):
            b.append(test[j][feature])
            break
        
      a.append(b)
      b = []

    
    return a


def generating_df_values(text_intro, list_of_tags):
  
  doc = nlp(text_intro)
  features_dict = print_pos_df(doc)

  text_list = []
  item_list = []

  for item in list_of_tags:
    item_list.append(item)
    text_list.append(text_intro)

  d = {'source': text_list, 'Tags' : item_list}
  dfx = pd.DataFrame(d)

  list_x = []
  for item in dfx['Tags'].to_list():
    list_x.append(item.split(' '))

  dfx['temp'] = list_x

  features = ['text', 'text_with_ws', 'lemma_', 'pos_', 'tag_', 'dep_', 'shape_', 'is_alpha', 'is_stop', 'ent_type_', 'ent_iob_']

  for feature in features:
    dfx[feature] = generate_features(feature, features_dict, list_x)
    # for i in dfx.index:
    #   dfx[feature][i] = ', '.join(str(x) for x in dfx[feature][i]) 

  return dfx





def pre_process(df):

	df['list_of_tags'] = df['list_of_tags'].apply(ast.literal_eval)

	cols = ['source', 'text', 'text_with_ws', 'lemma_', 'pos_', 'tag_', 'dep_', 'shape_', 'is_alpha', 'is_stop', 'ent_type_', 'ent_iob_', 'label_item']
	df2 = pd.DataFrame(columns= cols)

	ps = PorterStemmer()

	list_of_df = []

	for i in df.index:
		list_of_df.append(generating_df_values(df.text_intro[i], df.list_of_tags[i]))

	df3 = pd.concat(list_of_df)

	return df3




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--file_path", type=str)

	# read the arguments from the command line
	args = parser.parse_args()
	path = args.file_path

	df = pd.read_csv(path)

	dfinal = pre_process(df)

	dfinal.to_csv('pre_processed_data.csv')


	


