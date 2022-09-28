import pandas as pd
import ast
from nltk.stem import PorterStemmer
import inflect
import argparse


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # currently, we need path to dataset, flag to use a custom model, path to custom model, and flag to compute bias metrics
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--file_path_original", type=str)

    # read the arguments from the command line
    args = parser.parse_args()

    file_path=args.file_path
    file_path_original= args.file_path_original

    df = pd.read_csv(file_path)
    df1 = pd.read_csv(file_path_original, nrows = 5)

    df['gold_keys'] = df1['gold_keys'].apply(ast.literal_eval)
    df['topN_tags'] = df['topN_tags'].apply(ast.literal_eval)

    indexNames1 = df[ df['topN_tags'] == 'nan' ].index
    indexNames2 = df[ df['topN_tags'] == '[]' ].index
    df.drop(indexNames1 , inplace=True)
    df.drop(indexNames2 , inplace=True)
    df.drop(indexNames2 , inplace=True)

    for i in df.index:
      print(i , df['topN_tags'][i], type(df['topN_tags'][i]))

    ps = PorterStemmer()
    df['stemmed_output'] = 'nan'


    for i in df.index:
        manual_keywords = []
        for tag in df['topN_tags'][i]:
            manual_keywords.append(ps.stem(tag))

        df['stemmed_output'][i] = manual_keywords

    total_precision = 0
    total_recall = 0


    
    for idx in df.index:
        keywords = df['stemmed_output'][idx]
        print(len(keywords))

    
        manual_keywords = df['gold_keys'][idx]
    

        num_manual_keywords = len(manual_keywords)

        #correct = 0
        correct = sum(el in manual_keywords for el in keywords)
        print(correct) 

        total_precision += correct/float(len(keywords))
        total_recall += correct/float(num_manual_keywords)
        print('correct:', correct, 'out of', num_manual_keywords)
    

    print(len(df))
    avg_precision = round(total_precision*100/float(len(df)), 2)
    avg_recall = round(total_recall*100/float(len(df)), 2)

    avg_fmeasure = round(2*avg_precision*avg_recall/(avg_precision + avg_recall), 2)


    print("Precision", avg_precision, "Recall", avg_recall, "F-Measure", avg_fmeasure)


    

    



