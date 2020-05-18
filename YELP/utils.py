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

def labelize_reviews(reviews, label):
    for index, review in zip(reviews.index, reviews):
        yield TaggedDocument(review.split(), [label + '_%s' % index ])

def get_learned_vectors(model,corpus):
    """
    A function that extracts document vectors from a TRAINED Doc2Vec model
    
    model: Trained Doc2Vec model 
    """
    vecs = [model.docvecs['all_'+str(ind)] for ind, doc in corpus.iteritems()]
    
    return vecs

def labelize_reviews_bg(reviews, label, phraser):
    for index, review in zip(reviews.index, reviews):
        yield TaggedDocument(phraser[review.split()], [label + '_%s' % index ])