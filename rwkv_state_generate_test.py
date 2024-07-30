import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['OPS_KERNEL'] = '0'#是否使用纯py的wkv算子，0就使用cuda算子
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
tokenizer = RWKV_TOKENIZER(dict_path)
maxlen = 64
RWKV = build_transformer_model(
        config_path=config_path,
        model='rwkv6',
        keras_weights_path=weights_path,
        return_keras_model=False,
        sequence_length = maxlen,
        with_lm='softmax',
    )
rwkv = RWKV.model

n_chunk = 4
initial_states = None
test_text ='''\n下面是一个关于python实现冒泡排序'''

print('test generate')


generate_model = RWKV.build_cache_model(input_lengths=[maxlen],end_token=261,#rwkv的结束符号是\n\n，对应词表里就算261
                                        search_mode='topp',k=0.8,progress_print=False,index_bias=0,
                                        input_state=True,output_state=True)

def generate(test_text):
    x = tokenizer.encode(test_text)[0]
    x+= [0]*(maxlen-len(x))
    x = np.array([x],dtype='int32')
    x = ops.convert_to_tensor(x,dtype='int32')
    
    inputs = []
    for modelin in generate_model.inputs: 
        shape=keras.ops.shape(modelin)
        shape=[1 if t==None else t for t in shape]
        inputs.append(ops.zeros(shape,modelin.dtype))
    inputs[0] = x
    import time
    start = time.time()
    for _ in range(n_chunk):
        outs = generate_model.predict(inputs,verbose=3)
        out_ids = outs[0][0]
        print(tokenizer.decode([out_ids[out_ids!=0]])[0],end='')
        finally_token = out_ids[-1]
        ids = np.zeros([1,maxlen])
        ids[:,0] = finally_token
        inputs = [np.array(ids,'int32')]+outs[1:]

    times = time.time()-start
    print('\n')
    print('总耗时为'+str(times)+' 秒')
    print('推理速度为'+str((sum(out_ids!=0)+(n_chunk-1)*maxlen)//times)+' token/s')

generate(test_text)
generate(test_text)
