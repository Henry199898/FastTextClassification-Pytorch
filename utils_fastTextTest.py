# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000 # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>' # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
      构建一个词表：
      首先对数据集中的每一行句子按字/空格进行分割，然后统计所有元素的出现频率
      接下来按照频率从高到低的顺序对所有频率大于min_freq的元素进行排序，取前max_size个元素
      最后按照频率降序构建字典vocab_dic：{元素:序号}，vocab_dic的最后两个元素是'<UNK>'和'<PAD>'
    """
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        # tqdm是一个python的进度条工具包，在终端运行时显示程序循环的进度
        for line in tqdm(f):# 处理每一行
            lin = line.strip() # 移除头尾空格或换行符
            if not lin: # 跳过空行
                continue
            content = lin.split('\t')[0] # 句子和标签通过tab分割，前面的是句子内容，后面的是标签(lin.split('\t')[1])
            for word in tokenizer(content): # 按空格分割或者按字分割,tokenizer为可选择参数
                vocab_dic[word] = vocab_dic.get(word, 0) + 1 # 统计词频或字频，即为每个词或字在训练集中出现的次数
        # 遍历词典，筛选出词频大于min_freq=1的词，然后按照词频从高到低排序，取前max_size=10000个词，组成新的列表vocab_list，vocab_list中的元素为元组(word, freq)
        """
        sorted函数用法：sorted(iterable, cmp=None, key=None, reverse=False)
        iterable -- 可迭代对象。
        cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
        key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
        reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
        """
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 在vocab_dic的最后增加两个元素：{'<UNK>':len(vocab_dic)}和{'<PAD>':len(vocab_dic)+1}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    """
    加载数据集：
    对数据集中的每一行，先分离内容和标签
    然后对句子内容，按指定的方式进行分割（依照空格或字符），接着根据pad_size进行补足或截断
    接着把分割后的元素，通过词表转化成一串序号words_line
    最后把所有句子处理后的结果组成一个大列表，列表中的元素为：[(words_line, int(label), seq_len),...]
    """
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    # if os.path.exists(config.vocab_path):
    #     vocab = pkl.load(open(config.vocab_path, 'rb'))
    # else:
    # 此处使用训练集自己构建词表
    vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    # 构建完了之后保存为pickle
    pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f): # 打开数据文件，按行读取
                lin = line.strip() # 移除头尾空格或换行符
                if not lin: # 跳过空行
                    continue
                content, label = lin.split('\t') # 句子和标签通过tab分割，前面的是句子内容，后面的是标签
                words_line = []  # words_line是句子通过词表转化后得到的数字表示
                token = tokenizer(content) # 按空格或字符来分割句子
                seq_len = len(token)  # 得到分割后的元素数量
                if pad_size: # 如果有指定填充长度
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token))) #padding填充
                    else:
                        token = token[:pad_size] # 直接截断至填充长度
                        seq_len = pad_size  # 更新元素数量
                # word to id
                for word in token:
                    # 拿到该元素在词表中的序号，然后将这个序号添加到words_line中。如果该元素不在词表中，则填入'<UNK>'（未知字）的序号
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
                # 在contents中存入一个元组，元组的内容为（words_line，数字标签，元素数量，bigram，trigram）
                contents.append((words_line, int(label), seq_len, bigram, trigram))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    """
      根据数据集产生batch
      这里需要注意的是，在_to_tensor()中，代码把batch中的数据处理成了`(x, seq_len), y`的形式
      其中x是words_line，seq_len是pad前的长度(超过pad_size的设为pad_size)，y是数据标签
    """
    # 这里的batches就是经过build_dataset()中的load_dataset()处理后得到的contents：(words_line, int(label), seq_len, bigram, trigram)
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size # batch的容量（一次进多少个句子）
        self.batches = batches  # 数据集
        self.n_batches = len(batches) // batch_size # 数据集大小整除batch容量
        self.residue = False  # 记录batch数量是否为整数，false代表可以，true代表不可以，residuere是‘剩余物，残渣'的意思
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0 # 迭代用的索引
        self.device = device

    def _to_tensor(self, datas):
        # xx = [xxx[2] for xxx in datas]
        # indexx = np.argsort(xx)[::-1]
        # datas = np.array(datas)[indexx]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) # 句子words_line
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device) # 标签
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size，未超过的为原seq_size不变)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches: # 如果batch外还剩下一点句子，并且迭代到了最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)] # 直接拿出剩下的所有数据
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else: # 迭代器的入口，刚开始self.index是0，肯定小于self.n_batches
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size] # 正常取一个batch的数据
            self.index += 1
            batches = self._to_tensor(batches) # 转化为tensor
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):  # 这里的dataset是经过build_dataset()处理后得到的数据（vocab, train, dev, test）
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == "__main__":
    '''提取预训练词向量'''
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/vocab.embedding.sougou"
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
