from keras_nlp.models import BloomTokenizer
tokenizer = BloomTokenizer.from_preset("bloom_176b")
tokens = tokenizer("Hello 世界")
print(tokens)