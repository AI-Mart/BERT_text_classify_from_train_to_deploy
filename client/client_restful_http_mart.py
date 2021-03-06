# -*- coding: utf-8 -*-
"""
"""

import sys
sys.path.append('..')
import requests
import json
from bert import tokenization

tokenizer = tokenization.FullTokenizer(
        vocab_file='../roeberta_zh_L-12_H-768_A-12/vocab.txt',
        do_lower_case=True)

def text2ids(textList,max_seq_length):
    '''
    将输入的待分类文本编码为模型所需要的编码形式
    :parma textList: 输入的文本列表，形如 ['今天天气真好','我爱你']
    :return input_ids_list: 句子的向量表示形式
    :return input_mask_list: 只有一个句子，所以目前为固定格式
    '''
    input_ids_list = []
    input_mask_list = []
    for text in textList:
        tokens_a = tokenizer.tokenize(text)
            # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]


        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
    return input_ids_list, input_mask_list




def bert_class(textList):
    '''
    调用tfserving服务的接口，对外提供服务
    :parma textList: 输入的文本列表，形如 ['今天天气真好','我爱你']
    :return result: 结果
    '''
    input_ids_list, input_mask_list = text2ids(textList,440)
    url = 'http://127.0.0.1:8501/v1/models/versions:predict'#本地部署地址
    # url = 'http://10.20.xxx.88:8501/v1/models/versions:predict'#服务器部署地址
    data = json.dumps(
            {
                    "name": 'deeplab',
                    "signature_name":'result',
                    "inputs":{
                            'input_ids': input_ids_list,
                            'input_mask': input_mask_list}})
    result = requests.post(url,data=data).json()
    return result







if __name__ == '__main__':
    textList = ['''其他用房【第一次拍卖】上海市嘉定区嘉定工业区永盛路1218号107室、108室成交后，房产交付时的现状为准（本次拍卖不包括室内物品）；2、交易过程中产生的一切拍品介绍（1）房屋权属状况：上海市嘉定区嘉定工业区永盛路1218号107室（权证号：沪房地嘉字（2012）第018420，建筑面积：148.16平方米），非居住，登记日期：2012年09月。上海市嘉定区嘉定工业区永盛路1218号108室（权证号：沪房地嘉字（2012）第018419，建筑面积：122.07平方米），非居住，登记日期：2012年09月公告如下：一、拍卖标的：上海市嘉定区嘉定工业区永盛路1218号107室、108室起拍价:600万元，保证金：80万元，增价幅度：5万元。二、咨询、展示看样的时间与方式：自2019年10月28日起至2019年11月14日止（节假日休息）接受咨询。有意者统一定于2019年11月14日10:00—11:00时安排看样。三、本次拍卖活动设置延时出价功能，在拍卖活动结束前，每最后5分钟如果有竞买人出价，将自动延迟5分钟。四、拍卖方式：设''']
    result = bert_class(textList)
    print(result)

#只开启restfulAPI
"""
docker run --name tfserving-bert \
        --hostname tfserving-bert \
        -tid \
        --restart=on-failure:10 \
        -v  /etc/timezone:/etc/timezone \
        -v  /etc/localtime:/etc/localtime \
        -p 8501:8501 \
        -p 8502:8502 \
        --mount type=bind,source=/home/mart/versions,target=/models/versions \
        -e MODEL_NAME=versions \
        -t tensorflow/serving &
"""




#网页直接测试版本postman或者Google DEBUG-API
"""
http://10.20.xxx.88:8501/v1/models/versions:predict


 {
                    "name": "deeplab",
                    "signature_name":"result",
                    "inputs":{
                            "input_ids":  [[101, 1555, 689, 4500, 2791, 8024, 3636, 3727, 2356, 3736, 2279, 1277, 1227, 1220, 6125, 2458, 3209, 6662, 126, 1384, 7032, 1094, 1920, 1336, 712, 3517, 8123, 2231, 2791, 772, 8024, 2791, 2238, 4500, 6854, 1350, 1759, 1765, 2595, 6574, 117, 1215, 1062, 8024, 4385, 4307, 4500, 6854, 117, 1215, 1062, 8024, 3036, 2497, 5632, 6499, 6948, 2131, 928, 6397, 2099, 138, 8271, 140, 5018, 9247, 8818, 8159, 8177, 1384, 6397, 844, 2845, 1440, 8021, 2864, 1501, 1399, 4917, 3636, 3727, 2356, 3736, 2279, 1277, 1227, 1220, 6125, 2458, 3209, 6662, 126, 1384, 7032, 1094, 1920, 1336, 712, 3517, 8123, 2231, 2791, 772, 3326, 1164, 3341, 3975, 1385, 3791, 6161, 2137, 3326, 6395, 2658, 1105, 2791, 2238, 2792, 3300, 3326, 6395, 1384, 2791, 3326, 6395, 2279, 2099, 5018, 8151, 8279, 8756, 9153, 1384, 2864, 1501, 2792, 3300, 782, 6158, 2809, 6121, 782, 2864, 1501, 4385, 4307, 2791, 2238, 4500, 6854, 1350, 1759, 1765, 2595, 6574, 4385, 4307, 4500, 6854, 1215, 1062, 4909, 6595, 2658, 1105, 3300, 7170, 1267, 3187, 6981, 1947, 2658, 1105, 3717, 510, 4510, 510, 6858, 6380, 510, 4510, 3461, 510, 2160, 2372, 510, 3867, 7344, 5023, 8024, 6392, 3177, 6392, 1906, 7970, 1059, 511, 3326, 1164, 7361, 1169, 2658, 1105, 679, 6422, 2990, 897, 4638, 3152, 816, 122, 510, 517, 3791, 7368, 6161, 2137, 741, 518, 8039, 123, 510, 517, 1291, 1221, 2809, 6121, 6858, 4761, 741, 518, 8039, 124, 510, 517, 2864, 1297, 2768, 769, 8024, 3636, 3727, 2356, 3736, 2279, 1277, 1227, 1220, 6125, 2458, 3209, 6662, 126, 1384, 7032, 1094, 1920, 1336, 712, 3517, 8123, 2231, 2791, 772, 511, 3636, 3727, 2356, 3736, 2279, 1277, 4078, 7305, 6662, 2458, 3209, 6662, 769, 1349, 1366, 691, 1266, 6235, 8039, 2792, 1762, 2456, 5029, 4289, 1066, 8133, 2231, 8024, 1071, 855, 754, 5018, 8123, 2231, 8024, 676, 6956, 4510, 3461, 8039, 7167, 3921, 5310, 3354, 8024, 6205, 1298, 3308, 1403, 8024, 2456, 5029, 7481, 4916, 711, 8503, 8160, 119, 8409, 9236, 8039, 4385, 4307, 4500, 6854, 1215, 1062, 8024, 6421, 4289, 689, 2456, 5029, 4289, 5335, 2844, 671, 5663, 8024, 6858, 7599, 7023, 1045, 6772, 1962, 8024, 3187, 1765, 1825, 678, 3756, 1350, 1870, 860, 2458, 6162, 4638, 4385, 6496, 8024, 2456, 2768, 2399, 819, 5276, 8170, 2399, 8039, 2768, 3173, 4372, 8300, 110, 8024, 4385, 1139, 4909, 511, 2791, 2238, 2792, 3300, 3326, 6395, 1384, 2791, 3326, 6395, 2279, 2099, 5018, 8151, 8279, 8756, 9153, 1384, 8039, 1759, 1765, 6395, 1384, 2279, 1744, 4500, 8020, 8151, 8021, 5018, 10987, 1384, 8039, 886, 4500, 3326, 5102, 1798, 1139, 6375, 8039, 3326, 2247, 2595, 6574, 1744, 3300, 1759, 1765, 886, 4500, 3326, 8039, 4500, 6854, 1555, 102]],
                            "input_mask":  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}}

{
    "outputs": {
        "score": [
            [
                6.99151933e-05,
                0.999867678,
                6.23253363e-05
            ]
        ],
        "pred_label": [
            1
        ]
    }
}

"""
