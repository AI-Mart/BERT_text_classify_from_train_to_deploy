# -*- coding: utf-8 -*-
"""
"""
import sys
sys.path.append('..')
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

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
    # grpc配置
    channel = grpc.insecure_channel('127.0.0.1:8500')##本地部署地址
    # channel = grpc.insecure_channel('10.20.xxx.88:8500')  ##服务器部署地址
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
#查看接口信息 curl http://10.20.xxx.88:8501/v1/models/versions/metadata
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'versions'# """ 1: 模型的名称 """
    request.model_spec.signature_name = 'result'#此处的参数必须和生成pb文件save_model.py那个函数一一对应  """ 2: 服务的名称 """
    request.inputs['input_ids'].CopyFrom(tf.make_tensor_proto(input_ids_list))# """ 3: 输入的KEY """
    request.inputs['input_mask'].CopyFrom(tf.make_tensor_proto(input_mask_list))# """ 3: 输入的KEY """

    response = stub.Predict.future(request, 10.0)
    result = response.result()
    # score= tf.make_ndarray(result.outputs["score"])#""" 5: 输出的KEY """
    probs = tf.make_ndarray(result.outputs["pred_label"])#""" 5: 输出的KEY """
    return probs

if __name__ == '__main__':#当然可以多个数据注册batch_size 同时预测
    textList = ['''商业用房，武汉市江岸区劳动街开明路5号金冠大厦主楼18层房产，房屋用途及土地性质,公寓，现状用途,公寓，摘录自豫郑宏信评字[2018]第121194C号评估报告）拍品名称武汉市江岸区劳动街开明路5号金冠大厦主楼18层房产权利来源司法裁定权证情况房屋所有权证号房权证岸字第2012002608号拍品所有人被执行人拍品现状房屋用途及土地性质现状用途办公租赁情况有钥匙无配套情况水、电、通讯、电梯、宽带、消防等，设施设备齐全。权利限制情况不详提供的文件1、《法院裁定书》；2、《协助执行通知书》；3、《拍卖成交，武汉市江岸区劳动街开明路5号金冠大厦主楼18层房产。武汉市江岸区澳门路开明路交叉口东北角；所在建筑物共23层，其位于第18层，三部电梯；钢混结构，西南朝向，建筑面积为1049.78㎡；现状用途办公，该物业建筑物维护一般，通风采光较好，无地基下沉及墙体开裂的现象，建成年份约2009年；成新率85%，现出租。房屋所有权证号房权证岸字第2012002608号；土地证号岸国用（2012）第259号；使用权类型出让；权属性质国有土地使用权；用途商务金融用地；宗地面积112.99㎡；终止日期2043年6月2日；抵押信息已抵押；查封信息已查封；已出租。室内可自由分割，其装修装饰外墙面部分贴瓷片、部分为玻璃幕墙，入户门地弹簧玻璃门，室内部分为复合门带门套，部分为地弹簧玻璃门，铝合金窗，室内简单装修办公区域地面铺木地板，墙面刷乳胶漆，顶棚刷白；餐厅及卫生间地面铺地板砖，墙面下部贴瓷片，上部及顶棚刷白；走廊部位地面铺地板砖，墙面刷乳胶漆，顶棚吊顶；上下水管道畅通。配套设施水、电、通讯、电梯、宽带、消防等，设施设备齐全。起拍价1196.12万元，保证金120万元，增价幅度1万元。''']
    result = bert_class(textList)
    print(result)

#同时开启restfulAPI和GRPC
"""
docker run --name tfserving-bert \
        --hostname tfserving-bert \
        -tid \
        --restart=on-failure:10 \
        -v  /etc/timezone:/etc/timezone \
        -v  /etc/localtime:/etc/localtime \
        -p 8500:8500 \
        -p 8501:8501 \
        --mount type=bind,source=/home/mart/versions,target=/models/versions \
        -e MODEL_NAME=versions \
        -t tensorflow/serving &
"""

