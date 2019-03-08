from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim import models, corpora
import re
from nltk.corpus import brown
from gensim import similarities


data = []

for fileid in brown.fileids():
    document = ' '.join(brown.words(fileid))
    data.append(document)
# data = data[:10]
NO_DOCUMENTS = len(data)
print(NO_DOCUMENTS)
print(data[:5])


NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match(
        '[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in data:
    tokenized_data.append(clean_text(text))


# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# Have a look at how the 20th document looks like: [(word_id, count), ...]
print(corpus[20])
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...

# Build the LDA model
lda_model = models.LdaModel(
    corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

print("LDA Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

print("=" * 20)

text = "The economy is working better than ever"
bow = dictionary.doc2bow(clean_text(text))
print(lda_model[bow])

lda_index = similarities.MatrixSimilarity(lda_model[corpus])

# Let's perform some queries
similarities = lda_index[lda_model[bow]]
# Sort the similarities
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

# Top most similar documents:
print(similarities[:10])
# [(104, 0.87591344), (178, 0.86124849), (31, 0.8604598), (77, 0.84932965), (85, 0.84843522), (135, 0.84421808), (215, 0.84184396), (353, 0.84038532), (254, 0.83498049), (13, 0.82832891)]

# Let's see what's the most similar document
document_id, similarity = similarities[0]
print(data[document_id][:1000])
