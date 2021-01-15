import sys
from statistics import mean

results = {}
for i in range(1,11):
    with open('/data/naacl_2021_multilingual/naacl_eval/xlt/output_encoders_'+str(i)+'/te_scores_stackoverflow.csv') as f: 
        for line in f.readlines():
            if line[0] == ',':
                continue
            model_results = line.split(',')
            details = ','.join(model_results[:6])
            if details not in results:
                results[details] = {'precision':[], 'recall':[], 'f1':[]}
            results[details]['f1'].append(float(model_results[6]))
            results[details]['precision'].append(float(model_results[7]))
            results[details]['recall'].append(float(model_results[8]))

average_results = {}
for model in results:
    average_results[model] = {'f1':mean(results[model]['f1']), 'precision':mean(results[model]['precision']),'recall':mean(results[model]['recall'])}

with open('/data/naacl_2021_multilingual/naacl_eval/average_te_scores_xlt.csv','w') as f:
    f.write(',encoder,classifier,embedding_type,data,language,test f1 macro score,test precision macro score,test recall macro score\n')
    for model in average_results:
        f.write(model + ',' + str(average_results[model]['f1']) + ',' + str(average_results[model]['precision']) + ',' + str(average_results[model]['recall'])+'\n')

