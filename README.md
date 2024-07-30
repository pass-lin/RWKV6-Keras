# RWKV6-Keras
本模型主要基于bert4keras3构建。因此在阅读本文档前，请先阅读[rwkv的keras kernel实现](https://github.com/infiy-quine/RWKV6_Keras_Operator)和[bert4keras3实现](https://github.com/pass-lin/bert4keras3)，并根据对应的说明安装这两个依赖库。本实现由两个库的拥有者共同开发。  
模型的权重可以在[bert4keras3仓库](https://github.com/pass-lin/bert4keras3)中找到下载链接。我们会把所有的模型都上传到modelscope中方便高速下载。  
## 如何定义基于keras的rwkv模型 。 
```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['OPS_KERNEL'] = '0'#是否使用纯py的wkv算子，0就使用cuda算子
import keras
keras.config.set_floatx('bfloat16')#rwkv本身是一个bf16模型
from bert4keras3.models import build_transformer_model
from bert4keras3.tokenizers import RWKV_TOKENIZER
base_path = 'RWKV6-1.6B/'
config_path = base_path+'config.json'
weights_path = base_path+'model.weights.h5'
dict_path = base_path+'rwkv_vocab_v20230424.txt'
tokenizer = RWKV_TOKENIZER(dict_path)
RWKV = build_transformer_model(
        config_path=config_path,
        model='rwkv6',
        keras_weights_path=weights_path,
        return_keras_model=False,
        sequence_length = maxlen,
        with_lm='softmax',
    )#注意，这个RWKV是bert4keras3的一个类，不是一个keras模型
rwkv = RWKV.model#这个才是keras模型
rwkv.summary()#打印一下模型
#然后可以看一下rwkv，模型的输入和输出
print(rwkv.inputs)
print(rwkv.outputs)
print(tokenizer.encode(['hello world','我是中国人']))#输出一下rwkv tokenizer的结果
```
如果你使用的是tensorflow（个人并不建议）和jax（极度推荐）作为后端，建议maxlen设置为一个固定值。因为这俩会开启XLA加速，如果seqlen不是固定的，那每遇到一个新的seqlen都会做一次新的编译，从而导致速度特别慢。    
如果是torch或者mla作为后端，这里的maxlen可以设置为None。因为动态图下不会做编译其实就无所谓了。  
即使你是只做推理我也不建议使用numpy后端，因为我们实现的wkv纯python算子numpy使用是有问题的。所以如果你使用numpy作为推理后端，可能需要你自己去做一些修改。  

## 如何基于上面的模型训练RWKV 

```python
from bert4keras3.layers import Loss
from keras import ops
epochs = 10
steps_per_epoch = 100
class CrossEntropy(Loss):
        
    def compute_loss(self, inputs,mask=None):
        y_true, y_pred = inputs
        y_mask = ops.not_equal(y_true,0)
        y_pred = ops.cast(y_pred,'float32')
        y_true = ops.cast(y_true,'float32')
        y_mask = ops.cast(y_mask, y_pred.dtype)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = ops .sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        loss = ops.sum(loss * y_mask,-1) / (ops.sum(y_mask,-1)+1)
        
        return ops.mean(loss)
decay_steps = epochs*steps_per_epoch
initial_learning_rate = 5e-5
lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps)

output = CrossEntropy(-1,dtype='float32')(rwkv.inputs[:] + rwkv.outputs)
train_model = keras.models.Model(rwkv.inputs, rwkv.outputs)
train_model.compile(optimizer=keras.optimizers.AdamW(lr_decayed_fn,weight_decay=0.01,beta_2=0.95))

#假设你有一个已经处理好的x和y
x,y =load_data(...)
train_model.fit(x,y,epochs=epochs)
#当然你也可以用数据生成器，可以是自定义的生成器，可以是tf.data，也可以是torch.dataloader。
#下面我提供一个tf.data的演示
import tensorflow as tf#注意，除非后端是tensorflow，不然应该使用tensorflow-cpu
tf_files='train.tfrecords'#我假设从tfrecords中提取数据
raw_dataset = tf.data.TFRecordDataset(filenames =tf_files,num_parallel_reads=tf.data.experimental.AUTOTUNE)
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE      
feature_description = {
    'text': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True),
}
import tensorflow.experimental.numpy as tnp
def LeftPading(data):
    #因为大模型推理都要left pad，所以训练的时候干脆就一起left pad了吧
    tokens = data['text']
    index = int(tnp.min(tnp.sum(tf.equal(tokens,0),-1)))
    tokens = tf.concat([tokens[:,index:],tf.zeros([tf.shape(tokens)[0],index],tokens.dtype)],-1)    
    return {'Input-Token': tokens}
def _parse_function(example_proto):
  #解析函数，把tfrecords转化为tf的tensor
  parsed_features=tf.io.parse_single_example(example_proto, feature_description)
  y = parsed_features['text']
  return  {'text': x}

parsed_dataset = raw_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache().repeat()#这个是tf官网的教程
parsed_dataset = parsed_dataset.shuffle(500000)#定义一个缓冲池，大小是500000条数据，训练的时候会一边进一边出，出是 从这里面随机出的
parsed_dataset = parsed_dataset.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
parsed_dataset = parsed_dataset.padded_batch(batch_size)#因为tf不能左padding，所以要先右padding
parsed_dataset = parsed_dataset.map(LeftPading,num_parallel_calls=tf.data.experimental.AUTOTUNE)#然后再有我们的函数转化为左padding
train_model.fit(parsed_dataset,epochs=epochs)#开始训练
```
当然你也可以自己写一个dataloader，torch的也可以。也不一定是要返回一个字典，你返回一个list，然后shape和前面的rwkv.inputs一致就好了
## 如何开启state-tuning
说道rwkv的特色，就不得不提[state-tuning](https://zhuanlan.zhihu.com/p/695005541)了。  
在bert4keras3中我们提供了一种很简单的开启方法
```python
RWKV.enable_state_tunig(self,
        time_shitf_tuning=False
        )
```
rwkv原作者的文章里只对了time-mix层的wkv算子做了state-tuning。但其实rwkv接受来自上一时间的信息不只是wkv算子，还有两个time-shift层也可以接受上一时间的信息。都可以看作来自上一时间的state。  
因此我提供了time_shitf_tuning参数，如果设置为true则可以把time-shift也开启state-tuning。   
## 如何开启模型推理
 
简单来说首先我们和之前一样定义RWKV，当然我们还可以加入几个参数。
```python
penalty = 1.0
temperature = 1.0
RWKV = build_transformer_model(
        config_path=config_path,
        model='rwkv6',
        keras_weights_path=weights_path,
        return_keras_model=False,
        sequence_length = maxlen,
        with_lm='softmaxs' if penalty==1 and temperature ==1 else 'linear',
        penalty = penalty,
        penalty_window = None,
        max_penalty_range = None,
        temperature = temperature,
    )
```
with_lm='linear',如果是roformer或者bert使用unilm模型，则with_mlm='linear'    
penalty :生成模型的惩罚系数，可以参考https://blog.csdn.net/weixin_44826203/article/details/127495773.输入是1则不执行该参数  
penalty_window ：重复惩罚的窗口，假penalty_window=128，那对于一个1024长度的模型来说会分为8个窗口，每个token解码的时候针对当前窗口之前的token和上一个窗口做重复解码惩罚。如果是None，窗口相当于全部token。  
max_penalty_range ：重复惩罚的次数范围，输入是一个二维的list。比如输入是[2,5]，那么会统计窗口内的token出现次数.会对>=2的次数做惩罚,并且最大值为5  
temperature = 1.0：生成模型解码的温度  
这些参数在会在接下来的代理奏效  
```python
generate_model = build_cache_model(
    input_lengths,
    end_token,
    search_mode,
    k,
    progress_print,
    index_bias
    )
```
input_lengths：输入inputs的最大长度，是一个整数列表。长度与model.inputs是一致的，每个内容代表该输入的最大长度。  
例如rwkv的inputs，我可以这么设置
```python
maxlen=512
input_lengths=[maxlen]
```
end_token:解码结束的token，碰到这个token会提前结束  
search_mode：解码的方式，支持'greedy'、'topk'、'topp'三种  
k:search_mode是greedy时无效，是topk时应该是一个大于1的整数，是topp时应该是0-1的浮点数.在1.4.0版本，当使用topp的时候输入可以是一个二维list。如果输入是list，那么第一个数代表原来的p值，第二个数topk的k值。会先使用topk选择前k个再使用topp选择k个中概率的前p个。  
progress_print：在每个推理的step内是否展示进度条，只对torch后端有效  
如果output_state为true，那么模型除了输出生成结果，还会输出state。  
同理如果input_state为true，那么在输入的时候你还需要输入模型的state。  
这里的state指的是wkv层的state，两个time-shift层对应的state。
### RWKV模型一般化的推理
所谓的一般化推理，指的是和transformer一样，分为prefill和inference俩阶段。事实上我们获取的generate_model就已经把这俩阶段封装好了，那么接下来我们要做的其实就很简单了。
```python
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
```
这部分完整代码参考[rwkv推理代码](https://github.com/pass-lin/RWKV6-Keras/blob/main/rwkv_generate_test.py) 

### rwkv的state推理
state推理是rwkv的一个特色。transformer一般来说是不维护kv cache的，这是因为因为prefill占用时间相对来说会比较短，而kv cache很占用显存。当然，kimi这种长文本下prefill占用时间也很长，所以也会维护kv cache.  
但是rwkv不一样的点在于，他的state显存占用的常数级的。因此rwkv维护一个state相对来说会更容易。所以我们就提供了这部分的实现。  
具体代码如下所示，但是值得注意的是，这里我们的在构建generate model的时候input_state和output_state都设置为True。  
```python
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

```
值得注意的是，这里generate_model.predict输出的是一个numpy的array。这意味着这里面会有一个cpu到gpu的通信。如果你不介意显存可能会泄露，并且不希望由这部分通信的时间的话。代码可以修改为下面的样子 
```python
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
        outs = generate_model(inputs,tranning=False)
        out_ids = ops.convert_to_numpy(outs[0][0])#tokenizer不接受tensor作为输入
        print(tokenizer.decode([out_ids[out_ids!=0]])[0],end='')
        finally_token = out_ids[-1]
        ids = np.zeros([1,maxlen])
        ids[:,0] = finally_token
        inputs = [ops.convert_to_tensor(ids,'int32')]+outs[1:]

    times = time.time()-start
    print('\n')
    print('总耗时为'+str(times)+' 秒')
    print('推理速度为'+str((sum(out_ids!=0)+(n_chunk-1)*maxlen)//times)+' token/s')

generate(test_text)

``` 
这部分完整代码参考[rwkv state推理代码](https://github.com/pass-lin/RWKV6-Keras/blob/main/rwkv_state_generate_test.py) 

如果你想要知道模型的全部参数及其含义，你可以阅读[bert4keras3文档](https://github.com/pass-lin/bert4keras3/blob/main/document.md#bert4keras3layerslayernormsgroupnorm)