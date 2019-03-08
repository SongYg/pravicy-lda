import jieba
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
jieba.suggest_freq('桓温', True)


def cut_txt():
    files = os.listdir('./data')
    for i, ff in enumerate(files):
        with open(os.path.join('./data', ff), encoding='utf-8') as f,\
                open('./data/nlp_test{}.txt'.format(i*2+1), 'w', encoding='utf-8') as w:
            document = f.read()
            document_cut = jieba.cut(document)
            result = ' '.join(document_cut)
            w.write(result)


def read_txt():
    with open('./data/stop_words.txt', encoding='utf-8') as r:
        stpList = r.readlines()
        stpList = [w.strip() for w in stpList]

    txt = []
    txtFiles = ['./data/nlp_test1.txt',
                './data/nlp_test3.txt', './data/nlp_test5.txt']
    for tf in txtFiles:
        with open(tf, encoding='utf-8') as rf:
            rtxt = rf.read()
            txt.append(rtxt)
    return txt, stpList


def lda(txt, stpList):
    corpus = txt
    print(txt)
    cntVector = CountVectorizer(stop_words=stpList)
    cntTf = cntVector.fit_transform(corpus)
    print(cntTf.getnnz())
    lda_model = LatentDirichletAllocation(n_topics=2,
                                          learning_offset=50.,
                                          random_state=0)
    docres = lda_model.fit_transform(cntTf)
    print(docres)
    # print(lda_model.components_)


if __name__ == "__main__":
    # cut_txt()
    txt, stpList = read_txt()
    # print(txt)
    lda(txt, stpList)
