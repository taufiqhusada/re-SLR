import torch

from transformers import MarianMTModel, MarianTokenizer
import pickle
from tqdm import tqdm

file = open('/scratch2/thdaryan/re-SLR/dataset/anns/original/ref2/refcoco+/refs(unc)_de.p', 'rb')
object_pickle = pickle.load(file)
print(object_pickle[0])
print(len(object_pickle))

model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

src_text = [object_pickle[0]['sentences'][0]['sent']]

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
[tokenizer.decode(t, skip_special_tokens=True) for t in translated]

from nltk.tokenize import word_tokenize
token_text = word_tokenize('Zebra Kreatur Front und Zentrum', language='german')

for i in tqdm(range(len(object_pickle))):
    for j in range(len(object_pickle[i]['sentences'])):
        src_text = [object_pickle[i]['sentences'][j]['sent']]

        translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
        result_de = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
        result_de = result_de.lower()
        object_pickle[i]['sentences'][j]['raw'] = result_de
        object_pickle[i]['sentences'][j]['sent'] = result_de
        object_pickle[i]['sentences'][j]['tokens'] = word_tokenize(result_de, language='german')

print(object_pickle[0])
print(len(object_pickle))
with open('refcoco+_refs(unc)_de.p', 'wb') as handle:
    pickle.dump(object_pickle, handle)
