from sklearn.preprocessing import OneHotEncoder
import numpy as np

words = np.array([["cat"], ["dog"], ["fish"], ["cat"], ["dog"]])

encoder = OneHotEncoder(sparse_output=False)
encoded_words = encoder.fit_transform(words)
print(encoded_words)