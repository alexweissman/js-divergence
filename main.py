import bs4
import urlparse
import pprint
import nltk
import math
import operator
import re

#stopwords = ["the", "of","and", ",",".",":",";", "[", "]","i","a","in","my","to","with","!","'","''","--","?","his","is","footnote",
#             "``","'d","'s","be","for","me","not","that","as","have","he","him","it","this","thou","will","you","your"]

stopwords = [l.strip() for l in open('data/stopwords.txt').readlines()]
punctuation = [l.strip() for l in open('data/punctuation.txt').readlines()]

file1 = "data/hhguide.txt"
file2 = "data/beowulf.txt"

alpha = 0.5

# Add in punctuation, if desired
#stopwords += punctuation

# Return true if the string represents a number, false otherwise.
def isNumber(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# Computes the frequency distribution for a document.  This also replaces numbers and times with generic markers <NUMBER> and <TIME>.
def computeFreqDistribution(doc):
    #tokens = [nltk.word_tokenize(t) for t in nltk.sent_tokenize(doc)]
    tokens = nltk.regexp_tokenize(doc,'\S+')
    filtered_tokens = [w.lower().strip('.,?!"\'') for w in tokens]
    consolidated_tokens = []
    for w in filtered_tokens:
        #if re.match("[\d]+([/\.][\d]+)?$", w):
            #consolidated_tokens.append("<NUMBER>")
        #    continue
        if isNumber(w):
            consolidated_tokens.append("<NUMBER>")
            continue            
        elif re.match("[\d]+(pm|am)$", w):
            consolidated_tokens.append("<TIME>")
            continue
        elif re.match("[\d]+:[\d]+(pm|am)?$", w):
            consolidated_tokens.append("<TIME>")
            continue
        elif re.match("\(?(\w+)\)?$", w):
            m = re.match("\(?(\w+)\)?$", w)
            consolidated_tokens.append(m.group(1))
            continue
        else:
            consolidated_tokens.append(w) 
        
    consolidated_tokens = [w for w in consolidated_tokens if not w in stopwords and w != "" ]
             
    #filtered_tokens = [w.lower().strip(''.join(punctuation)) for w in tokens if not w.lower().strip(''.join(punctuation)) in stopwords and w.lower().strip(''.join(punctuation)) != ""]
    fd = nltk.FreqDist(consolidated_tokens)
    return fd

# Computes the relative frequencies of the most common unigrams in a document.  Use the `limit` parameter to specify how many unigrams to compute.  Returns a dictionary of words -> relative frequencies.
def computeUnigramDistribution(doc, limit = 100):
    fd = computeFreqDistribution(doc)
    keys = fd.keys()[:limit]
    values = fd.values()[:limit]
    N = float(sum(values))
    dist = {}
    for key in keys:
        dist[key] = float(fd[key])/N
    return (dist,N)

# Computes the relative frequency distribution within a pair of documents.  Use the `limit` parameter to specify how many unigrams to compute.  Returns a dictionary of words -> relative frequencies.
def mergeDistribution(doc1, doc2, limit = 100):
    fd1 = computeFreqDistribution(doc1)
    keys = fd1.keys()[:limit]
    # First, copy the counts from the first distribution
    mergeCounts = {}
    for key in keys:
        mergeCounts[key] = fd1[key]
    # Now, add the counts from the second distribution
    fd2 = computeFreqDistribution(doc2)
    keys = fd2.keys()[:limit]
    for key in keys:
        if key in mergeCounts.keys():
            mergeCounts[key] += fd2[key]  
        else:
            mergeCounts[key] = fd2[key]
      
    values = mergeCounts.values()[:limit]
    N = float(sum(values))
    dist = {}
    for key in keys:
        dist[key] = float(mergeCounts[key])/N
    return (dist, N)

def mergeDistributionJS(dist1, dist2, alpha):
    mergeDist = {}
    for key in dist1.keys():
        mergeDist[key] = alpha*dist1[key]
    for key in dist2.keys():
        if key in mergeDist.keys():
            mergeDist[key] += (1-alpha)*dist2[key]
        else:
            mergeDist[key] = (1-alpha)*dist2[key]
    return mergeDist

def computeEntropy(dist):
    ent= 0
    for prob in dist.itervalues():
        ent += prob*math.log(prob,2)
    return -1*ent

def computeWordEntropy(dist, word):
    if word in dist.keys():
        return -1*dist[word]*math.log(dist[word],2) - (1-dist[word])*math.log((1-dist[word]),2)
    else:
        return 0

def computeWordJSDivergence(dist1, dist2, word, alpha):
    distBoth = mergeDistributionJS(dist1, dist2, alpha)
    h1 = computeWordEntropy(dist1, word)
    h2 = computeWordEntropy(dist2, word)
    hBoth = computeWordEntropy(distBoth, word)
    return hBoth - alpha*h1 - (1-alpha)*h2

doc1 = open(file1, 'r').read()
doc2 = open(file2, 'r').read()

(dist1, N1) = computeUnigramDistribution(doc1, 1000)

#pprint.pprint(dist1)

(dist2, N2) = computeUnigramDistribution(doc2, 1000)

#pprint.pprint(dist2)

p1 = float(N1)/float(N1+N2)

p2 = float(N2)/float(N1+N2)

distBoth = mergeDistributionJS(dist1, dist2, alpha)

#pprint.pprint(distBoth)

ent1 = computeEntropy(dist1)
ent2 = computeEntropy(dist2)
entBoth = computeEntropy(distBoth)

js = entBoth - alpha*ent1 - (1-alpha)*ent2

print "Document entropy"
print "----------------"
print "H(%s) = %f", (file1, ent1)
print "H(%s) = %f", (file2, ent2)
print "H(both) = ", entBoth
print "JS divergence = ", js

# Now, figure out the divergence between the distributions, based solely on each particular word
divergences = {}
odds = {}
guessDoc1 = []
guessDoc2 = []
dist1Unique = []
dist2Unique = []
for word in distBoth.keys():
    if (word in dist1.keys()) and (word in dist2.keys()):
        divergences[word] = computeWordJSDivergence(dist1, dist2, word, alpha)
        #odds[word] = math.log(dist1[word]*p1/distBoth[word]/(dist2[word]*p2/distBoth[word]),2)
        odds[word] = math.log(dist1[word]/dist2[word])
    else:
        if word in dist1.keys():
            dist1Unique.append(word)
        if word in dist2.keys():
            dist2Unique.append(word)        

divergences_sorted = sorted(divergences.iteritems(), key=operator.itemgetter(1), reverse=True)

print "--------- Top non-unique, distinguishing words --------"
for (word, div) in divergences_sorted[:200]:
    print word, " div = ", div, " Odds of ", file1, " to ", file2, " = ", odds[word], " [%s]" % (file1 if odds[word] > 0 else file2)
    # Since we're computing odds, also make a guess as to which words most likely came from one document vs the other
    if odds[word] > 0:
        guessDoc1.append(word)
    else:
        guessDoc2.append(word)
print "--------- Unique words from %s ---------" % file1
print ", ".join(sorted(dist1Unique))
print "--------- Unique words from %s ---------" % file2
print ", ".join(sorted(dist2Unique))
print "--------- Likely words from %s ---------" % file1
print ", ".join(sorted(guessDoc1))
print "--------- Likely words from %s ---------" % file2
print ", ".join(sorted(guessDoc2))
