import pandas as pd
import urllib3
import json
import time
import numpy

data=pd.read_csv("C:\\Users\\Shen Tian\\Desktop\\聊天机器人-情感极性.csv",encoding='ansi')
review_list=data.review

labels=[]
label_prediction =[]

access_token='24.fe2c425d63d97154e005d15ea0ce0dae.2592000.1775017840.282335-122190875'
http=urllib3.PoolManager()
url='https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token='+access_token

for i in range(len(review_list)):
    if (i + 1) % 2 == 0:
        time.sleep(1)
    params={'text':review_list[i]}
    encoded_data = json.dumps(params).encode('utf-8')
    request = http.request('POST',
                           url,
                           body=encoded_data,
                           headers={'Content-Type': 'application/json'})
    result = str(request.data, 'utf-8')
    a = json.loads(result)
    output = a['items'][0]
    labels.append(output['sentiment'])
    label_prediction.append(output['positive_prob'])
    print("第 " + str(i + 1) + "条评论已完成分析")
    

data['label']=labels
data['positive_prob']=label_prediction
data.to_csv("C:\\Users\\Shen Tian\\Desktop\\聊天机器人-情感极性-result.csv",encoding="utf-8")
