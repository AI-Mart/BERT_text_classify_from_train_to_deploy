python版本为3.6.7\
第一部分：各个文件介绍\
1、bert文件夹:google官方bert的源代码\
2、client文件夹：模型部署后给客户端调用的接口文件\
3、data文件夹：模型训练数据集,放了少量的数据方便对战数据结构，实际训练数据大概13万条，大小50M左右的数据集\
4、model_deploy_classify文件夹：模型训练完成后运行save_model_to_pb_mart.py生成的模型部署pb文件\
5、output_model文件夹：运行run_classifier.py后存储训练完成的模型文件\
6、roeberta_zh_L-12_H-768_A-12文件夹：google开源的中文预训练模型\
7、README.md文件：说明介绍文件\
8、requirements文件：项目运行的环境打包说明\
9、run_classifier.py文件：训练模型直接运行此文件\
10、save_model_to_pb_mart.py文件：模型训练完成后运行此文件可以生成模型部署文件\

第二部分：训练模型\
1、下载预训练模型，网盘链接：https://pan.baidu.com/s/1yPD5sf-3c4iE-7zrkg-UYg 提取码：5d5e ，解压后把文件夹里面的ckpt等预训练文件放在工程的roeberta_zh_L-12_H-768_A-12文件夹\
2、安装工程运行环境，打开终端一键运行安装 pip install -i https://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com  -r requirement.txt\
3、运行run_classifier.py模型训练，超参数配置如下，也可以按需求把data文件夹的数据按一样的格式换成自己的数据，训练完的模型在工程的output_model文件夹\
3.0、训练train:\
python run_classifier.py \
--task_name=sim \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=data \
--vocab_file=roeberta_zh_L-12_H-768_A-12/vocab.txt \
--bert_config_file=roeberta_zh_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=roeberta_zh_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=440 \
--train_batch_size=4 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=output_model\
3.1、模型训练完成后，也可以运行run_classifier.py按如下配置进行验证操作\
验证eval:\
python run_classifier.py \
--task_name=sim \
--do_eval=true \
--data_dir=data \
--vocab_file=roeberta_zh_L-12_H-768_A-12/vocab.txt \
--bert_config_file=roeberta_zh_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=output_model \
--max_seq_length=440 \
--output_dir=output_model\
3.2、模型训练完成后，也可以运行run_classifier.py按如下配置进行预测操作\
预测predict:\
python run_classifier.py \
--task_name=sim \
--do_predict=true \
--data_dir=data \
--vocab_file=roeberta_zh_L-12_H-768_A-12/vocab.txt \
--bert_config_file=roeberta_zh_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=output_model \
--max_seq_length=440 \
--predict_batch_size=4 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--output_dir=output_model\
4、最终模型的准确率如下\
***** Eval results output_model\model.ckpt-92371 *****\
eval_accuracy = 0.99976456\
eval_f1 = 0.99994093\
eval_loss = 0.0016948085\
eval_prec = 0.99997854\
eval_recall = 0.9999034\
global_step = 92371\
loss = 0.0016947261\

第三部分：本地win10系统部署和调用模型\
1、运行工程的save_model_to_pb_mart.py文件，会直接生成模型部署文件，并存放在model_deploy_classify文件夹\
2、按照此链接按照docker到win10系统，https://www.runoob.com/docker/windows-docker-install.html\
3、打开win10系统PowerShell运行指令\
4、输入指令 docker pull tensorflow/serving:1.14.0\
5、输入指令 docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=E:\AI_train\Z1_CLS\Z_AUCTION_CLS_TF14\mart21_github_RoBERTa_12_roberta_zh\model_deploy_classify/versions,target=/models/versions -e MODEL_NAME=versions -t tensorflow/serving:1.14.0\
其中source为模型文件夹所在电脑的绝对路径\
6、可运行docker ps -a，查看部署情况\
7、以上代码运行后没有报错，部署成功后可以运行client文件夹的文件进行模型调用，默认的地址是本地部署地址，两种方式调用分别是grpc和http\

第四部分：linux服务器部署和调用模型\
1、服务器xshell直接运行部署模型: 服务器注意python的名字对应的版本号,把model_deploy_classify文件压缩后，通过rz指令上传到服务器，然后通过unzip解压，通过rm删除多余的文件，通过ls查看当前目录文件，通过pwd查看当前目录，注意切换虚拟环境，服务器一定是python3去运行，默认python是python2的，一定要做好运行区分\
2、直接在服务器运行以下指令，注意source为模型在服务器的绝对路径\
docker run --name tfserving-bert \
        --hostname tfserving-bert \
        -tid \
        --restart=on-failure:10 \
        -v  /etc/timezone:/etc/timezone \
        -v  /etc/localtime:/etc/localtime \
        -p 8500:8500 \
        -p 8501:8501 \
        --mount type=bind,source=/home/mart/model_deploy_classify/versions,target=/models/versions \
        -e MODEL_NAME=versions \
        -t tensorflow/serving &
3、部署成功后可以运行client文件夹的文件进行模型调用，默认的地址是本地部署地址，把IP改成服务器的IP即可直接调用，两种方式调用分别是grpc和http\
