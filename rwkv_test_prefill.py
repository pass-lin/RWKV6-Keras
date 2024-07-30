import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['OPS_KERNEL'] = '0'#1使用纯py的wkv算子，0就使用cuda算子
import keras
keras.config.set_floatx('bfloat16')
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import RWKV_TOKENIZER
from bert4keras3.Models.RWKV import *
import numpy as np
base_path = 'RWKV6-1.6B/'
config_path = base_path+'config.json'
weights_path = base_path+'model.weights.h5'
dict_path = base_path+'rwkv_vocab_v20230424.txt'
maxlen = 1024
tokenizer = RWKV_TOKENIZER(dict_path)
rwkv = build_transformer_model(
        config_path=config_path,
        model='rwkv6',
        keras_weights_path=weights_path,
        return_keras_model=True,
        sequence_length = maxlen,
        with_lm='softmax',
    )
x = np.ones([64,1024])#随机生成，只测试速度用
rwkv.predict(x,batch_size = 4)
