from gensim.models.doc2vec import TaggedDocument

def clean_preprocess(text):
    """
    Given a string of text, the function:
    1. Remove all punctuations and numbers
    2. Converts texts to lowercase
    3. Handles negation words defined above.
    3. Returns cleaned text

    Parameters
    ----------
    text: str
        A string to be pre-processed
    """
    lower = re.sub(r'[^a-zA-Z\s\']', "", text).lower()
    lower_neg_handled = n_pattern.sub(lambda x: n_dict[x.group()], lower)
    letters_only = re.sub(r'[^a-zA-Z\s]', "", lower_neg_handled)
    words = [i for i  in tok.tokenize(letters_only) if len(i) > 1]
    return (' '.join(words))

def tokenize(review):
    """
    Get tokens for each review in the dataframe. 

    Parameters
    ----------
    review:
        A pandas series containing the reviews.

    Returns
    ----------
    A list of tokenized words from each review.  
    """
    tokens = []
    for i in tqdm(range(len(review.values))):
        tokens.append(clean_preprocess(review.values[i]))
    return tokens


def labelize_reviews(reviews,label):
    result = []
    prefix = label
    for i, j in zip(reviews.index,reviews):
        result.append(TaggedDocument(j.split(),[prefix + '_%s' % i ]))
    return result