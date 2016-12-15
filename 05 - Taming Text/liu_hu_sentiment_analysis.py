# liu hu lexicon sentiment analysis
# adapted from http://www.nltk.org/_modules/nltk/sentiment/util.html - demo_liu_hu_lexicon
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank
tokenizer = treebank.TreebankWordTokenizer()

def sentiment_liu_hu(text):
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(text)]

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
        elif word in opinion_lexicon.negative():
            neg_words += 1

    if pos_words > neg_words:
        return 1
    elif pos_words < neg_words:
        return -1
    elif pos_words == neg_words:
        return 0

# assigning only positive and negative may be too rash, we should somehow take into account the number 
# of words and the difference between the number of positive words and the number of negative words
def sentiment_liu_hu_mod(text):
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(text)]

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
        elif word in opinion_lexicon.negative():
            neg_words += 1
            
    return (pos_words - neg_words)/len(tokenized_sent)

# load from pickle
countries_dict = pickle.load( open( "countries_dict.p", "rb" ) )

country_sentiment_liu = {}
country_sentiment_liu_mod = {}
country_count = {}
email_num = 1
num_emails = len(emails.ExtractedBodyText)

# Step 1 - compute cumulative scores and number of mentions
for text in emails.ExtractedBodyText:
    
    #debug info
    if email_num % 1000 == 0:
        print("Email number %d/%d" % (email_num,num_emails))
    email_num += 1
    
    if text is not np.nan:  # skip text if invalid
        # split text into lines
        lines_list = tokenize.sent_tokenize(text)
        # for each line search for countries and perform sentiment "analysis"
        for line in lines_list:
            countries_found = None
            countries_found = list(set([countries_dict[c] for c in countries_dict.keys() if c in line]))
            if countries_found is not None:  # if found country, perform sentiment analysis
                score = sentiment_liu_hu_mod(line)
                # update score for each country
                for country in countries_found:
                    try:
                        country_sentiment_liu_mod[country] += score
                        country_sentiment_liu[country] += np.sign(score)
                        country_count[country] += 1
                    except:  # if country not yet in dictionary
                        country_sentiment_liu_mod[country] = score
                        country_sentiment_liu[country] = np.sign(score)
                        country_count[country] = 1