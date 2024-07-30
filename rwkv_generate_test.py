import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['OPS_KERNEL'] = '1'#1是否使用纯py的wkv算子，0就使用cuda算子
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
maxlen = 2048
tokenizer = RWKV_TOKENIZER(dict_path)
RWKV = build_transformer_model(
        config_path=config_path,
        model='rwkv6',
        keras_weights_path=weights_path,
        return_keras_model=False,
        sequence_length = maxlen,
        with_lm='softmax',
    )

rwkv = RWKV.model

generate_model = RWKV.build_cache_model(input_lengths=[maxlen],end_token=-1,#rwkv的结束符号是\n\n，对应词表里就算261
                                        search_mode='topp',k=0.5,progress_print=True,index_bias=0)
generate_model.compile( jit_compile="auto",)

text = '''\n下面是一个关于python实现'''
print('test generate')
def generate(text):
    x = tokenizer.encode(text)[0]
    x+= [0]*(maxlen-len(x))
    x = np.array([x],dtype='int32')
    import time
    start = time.time()
    o2 = generate_model.predict([x])[0]
    o2 = o2[o2!=0]
    print(len(o2)-len(tokenizer.encode(text)[0]))
    print(tokenizer.decode([o2])[0])
    times = time.time()-start
    print('\n')
    print('总耗时为'+str(times)+' 秒')
    print('推理速度为'+str(sum(o2!=0)//times)+' token/s')

generate(text)
generate(text)