import argparse
import copy
import logging
import numpy as np
import re
import os
import sys

from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger('sumGram.sumgram')

def get_dual_stopwords(add_stopwords):
    return add_stopwords | getStopwordsSet()

def bifurcate_stopwords(add_stopwords):
    
    unigrams = set()
    multigrams = set()

    for word in add_stopwords:

        wlen = len(word.split(' '))
        if( wlen == 1 ):
            unigrams.add(word)
        else:
            multigrams.add(word)

    return {
        'unigrams': unigrams,
        'multigrams': multigrams
    }


def combine_ngrams(ngram_lst):

    ngram_sentence = []

    for ngram in ngram_lst:
        ngram_sentence.append( set(ngram['ngram'].lower().split(' ')) )

    return ngram_sentence

def get_ngram_dct(ngram, tf, postings, extra_fields=None):

    if( extra_fields is None or isinstance(extra_fields, dict) == False ):
        extra_fields = {}

    payload = {'ngram': ngram, 'term_freq': tf}
    if( postings is not None ):
        payload['postings'] = postings
    
    for key, val in extra_fields:
        payload[key] = val

    return payload

def fmt_posting(doc_dets):
    dets_cp = copy.deepcopy(doc_dets)

    if( 'text' in dets_cp ):
        del dets_cp['text']

    if( 'doc_id' in dets_cp ):
        del dets_cp['doc_id']

    return dets_cp

def fmt_report(ngram_lst, params):

    params['add_stopwords'] = list( params['add_stopwords'] )
    params['add_stopwords_file'] = list( params['add_stopwords_file'] )

    for i in range(len(ngram_lst)):
        
        if( 'sumgram_history' in ngram_lst[i] ):
            ngram_lst[i]['base_ngram'] = ngram_lst[i]['sumgram_history'][0]['prev_ngram']

        if( params['include_postings'] is False ):
            ngram_lst[i].pop('postings', None)

        if( params['no_parent_sentences'] is True ):
            ngram_lst[i].pop('parent_sentences', None)

def calc_avg_sentence_overlap(ngram_sentences, doc_sentence):

    ngram_sent_size = len(ngram_sentences)
    if( ngram_sent_size == 0 or len(doc_sentence) == 0 ):
        return 0

    ov = 0
    for ngram_sent in ngram_sentences:
        ov += overlapFor2Sets(ngram_sent, doc_sentence)

    return ov/ngram_sent_size

def get_docs_sentence_score(ngram_sentences, sentences, doc_indx, doc_id, params):

    if( len(sentences) == 0 ):
        return []

    sentences_lst = []
    for i in range(  len(sentences) ):
        
        sentence = sentences[i]['sentence'].strip()
        if( sentence == '' ):
            continue

        if( 'avg_overlap' in sentences[i] ):
            #used to avoid calculating overlap for bad (sentences that are too long and could not be split by ssplit or regex) sentences
            ov = sentences[i]['avg_overlap']
        else:
            #ensure the splitting pattern corresponds to that used for ngrams
            sent_set = set( re.findall(params['token_pattern'], sentence.lower()) )
            ov = calc_avg_sentence_overlap( ngram_sentences, sent_set )
        
        sent = {
            'avg_overlap': ov,
            'sentence': sentence,
            'doc_indx': doc_indx,
            'doc_id': doc_id,
            'sent_indx': i
        }

        for key, val in sentences[i].items():
            if( key == 'sentence' ):
                continue
            sent[key] = val

        sentences_lst.append(sent)
    
    return sentences_lst

def parallel_nlp_add_sents(doc_dct_lst, params):

    size = len(doc_dct_lst)
    if( size == 0 or params['stanford_corenlp_server'] is False or params['sentence_tokenizer'] != 'ssplit'):
        return doc_dct_lst
    
    params.setdefault('thread_count', 5)
    params.setdefault('update_rate', 50)
    params.setdefault('corenlp_port', '9000')
    params.setdefault('corenlp_host', 'localhost')
    
    jobs_lst = []
    for i in range(size):

        if( i % params['update_rate'] == 0 ):
            print_msg = 'segmenting sentence i: ' + str(i) + ' of ' + str(size)
        else:
            print_msg = ''

        keywords = {
            'text': doc_dct_lst[i]['text'].replace('\n', ' '),
            'host': params['corenlp_host'],
            'port': params['corenlp_port']
        }
        
        jobs_lst.append({
            'func': nlpSentenceAnnotate, 
            'args': keywords, 
            'misc': False,
            'print': print_msg
        })
    
    res_lst = parallelTask(jobs_lst, threadCount=params['thread_count'])
    for i in range(size):
        if( 'sentences' in res_lst[i]['output'] ):
            doc_dct_lst[i]['sentences'] = res_lst[i]['output']['sentences']

    return doc_dct_lst

def extract_doc_sentences(doc, sent_tokenizer_pattern, dedup_set, multi_word_proper_nouns, params):

    filtered_sentences = []
    if( 'sentences' in doc ):
        if( len(doc['sentences']) != 0 ):
            for sent in doc['sentences']:

                sentence = sent['sentence'].replace('\n',  ' ').strip()
                lowercase_sent = sentence.lower()
                if( sentence == '' ):
                    continue
                
                if( lowercase_sent in dedup_set ):
                    continue

                tok_len = len(sent['tokens'])
                dedup_set.add(lowercase_sent)
                
                extract_proper_nouns( sent['tokens'], multi_word_proper_nouns )
                extract_collocation_cands( sent['tokens'], multi_word_proper_nouns, params )
                
                if( tok_len > params['corenlp_max_sentence_words'] ):
                    #this sentence is too long so force split 
                    filtered_sentences += regex_get_sentences(sent['sentence'], sent_tokenizer_pattern, dedup_set, params['corenlp_max_sentence_words'])
                else:
                    filtered_sentences.append({ 'sentence': sentence, 'segmenter': 'ssplit' })

    #if corenlp sentence segmentation is not used, use regex sentence segmentation
    if( len(filtered_sentences) == 0 ):
        filtered_sentences = regex_get_sentences(doc['text'], sent_tokenizer_pattern, dedup_set, params['corenlp_max_sentence_words'])

    return filtered_sentences

def regex_get_sentences(text, sent_tokenizer_pattern, dedup_set, corenlp_max_sentence_words):

    if( text == '' ):
        return []

    sentences = re.split(sent_tokenizer_pattern, text)
    filtered_sentences = []

    for sentence in sentences:
        
        sentence = sentence.replace('\n', ' ').strip()
        lowercase_sent = sentence.lower()
        if( sentence == '' ):
            continue
        
        if( lowercase_sent in dedup_set ):
            continue
    
        dedup_set.add(lowercase_sent)
        sent = { 'sentence': sentence, 'segmenter': sent_tokenizer_pattern }

        if( len(sentence.split(' ')) > corenlp_max_sentence_words ):
            sent['avg_overlap'] = -1

        filtered_sentences.append(sent)

    return filtered_sentences


def get_ranked_docs(ngram_lst, doc_dct_lst):

    ranked_docs = {}
    doc_id_new_doc_indx_map = {}
    N = len(ngram_lst)

    for i in range( len(ngram_lst) ):

        for posting in ngram_lst[i]['postings']:
            
            '''
                Give credit to documents that have highly ranked (bigger diff: N - i) terms in the ngram_lst 
                a document's score is awarded by accumulating the points awarded by the position of terms in the ngram_lst.
                Documents without terms in ngram_lst are not given points
            '''
            doc_indx = posting['doc_indx']
            
            ranked_docs.setdefault( doc_indx, {'score': 0, 'doc_id': doc_dct_lst[doc_indx]['doc_id'], 'doc_details': fmt_posting( doc_dct_lst[doc_indx] )} )
            ranked_docs[doc_indx]['score'] += (N - i)

    ranked_docs = sortDctByKey(ranked_docs, 'score')
    
    for i in range( len(ranked_docs) ):
        doc_id = ranked_docs[i][1]['doc_id']
        doc_id_new_doc_indx_map[doc_id] = i
    
    return ranked_docs, doc_id_new_doc_indx_map

def rank_sents_frm_top_ranked_docs(ngram_sentences, ranked_docs, all_doc_sentences, extra_params=None):

    if( extra_params is None ):
        extra_params = {}

    extra_params.setdefault('sentences_rank_count', 20)
    all_top_ranked_docs_sentences = []
    
    for doc in ranked_docs:
        
        doc_indx = doc[0]
        doc_id = doc[1]['doc_id']
        all_top_ranked_docs_sentences += get_docs_sentence_score(ngram_sentences, all_doc_sentences[doc_indx], doc_indx, doc_id, extra_params)
    
    return sorted(all_top_ranked_docs_sentences, key=lambda x: x['avg_overlap'], reverse=True)[:extra_params['sentences_rank_count']]


def interpolate_toks(span, group, pos_tok_map):

    if( len(span) != 2 or len(pos_tok_map) == 0 ):
        return {}

    pos_seq = []
    tok_seq = []
    for i in range(span[0], span[1]):
        
        if( i in pos_tok_map ):
            pos_seq.append( pos_tok_map[i]['pos'] )
            tok_seq.append( pos_tok_map[i]['tok'] )
        
    if( ' '.join(pos_seq) == group ):
        return {'pos': pos_seq, 'toks': tok_seq}
    else:
        return {}

def extract_collocation_cands(sent_toks, container, params):
    
    if( len(sent_toks) < 2 ):
        return

    '''
        Rules inspired by: https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a
        Bigrams:
            (Noun, Noun), (Adjective, Noun)
            NN[^ ]? NN[^ ]?S?
            JJ[^ ]? NN[^ ]?S?
        Trigrams:
            (Adjective/Noun, Anything, Adjective/Noun)
            (JJ[^ ]?|NN[^ ]?S?) \w+ (JJ[^ ]?|NN[^ ]?S?)
    '''

    POS = []
    pos_tok_map = {}#responsible for mapping position of POS to toks
    sequence = ''
    
    adj = 'JJ[^ ]?'
    w = ' \w+ '
    nn = 'NN[^ ]?S?'

    params['collocations_pattern'] = params['collocations_pattern'].strip()
    if( params['collocations_pattern'] == '' ):
        '''
            Switched off because benefit was not found proportional to cost. It splits multi-word proper nouns even though
            it also includes unsplit version, thus it returns large sets.
            Could be used instead of extract_proper_nouns() with rule: "NNP ((IN|CC)? ?NNP)+" but I advise against it because pattern matching is expensive
        '''
        return
        #rules inspired by 
        bigram_collocations = 'NN[^ ]? NN[^ ]?S?|JJ[^ ]? NN[^ ]?S?'
        trigram_collocations = adj + w + adj + '|' + adj + w + nn + '|' + nn + w + adj + '|' + nn + w + nn
        collocs = [bigram_collocations, trigram_collocations]
    else:
        collocs = [ params['collocations_pattern'] ]
    
    cursor = sent_toks[0]['pos']
    POS.append( sent_toks[0]['pos'] )
    pos_tok_map[0] = {'pos': sent_toks[0]['pos'], 'tok': sent_toks[0]['tok']}


    #sent_toks example: NNP VBZ JJ TO VB DT JJ JJ JJ NN IN NNP NN : NNP NNP NNP NNP VBZ JJ TO VB DT JJ JJ JJ NN IN NNP NN DT VBP DT CD RBS JJ JJ NNS IN NNP NN , VBG TO NN .
    for i in range( 1, len(sent_toks) ):
        
        POS.append( sent_toks[i]['pos'] )
        
        cursor = cursor + ' '
        pos_tok_map[ len(cursor) ] = {'pos': sent_toks[i]['pos'], 'tok': sent_toks[i]['tok']}
        cursor = cursor + sent_toks[i]['pos']
    

    POS = ' '.join(POS)
    for pattern in collocs:
        
        p = re.compile(pattern)
        for m in p.finditer(POS):

            collocation = interpolate_toks( m.span(), m.group(), pos_tok_map )
            if( len(collocation) == 0 ):
                continue
            
            colloc_text = ' '.join( collocation['toks'] )
            collocation_lower = colloc_text.lower()

            #consider accounting for NNPS and possible NN in calculating proper_noun_rate
            proper_noun_rate = round( collocation['pos'].count('NNP')/len(collocation['pos']), 4 )
            
            if( collocation_lower in container ):
                container[collocation_lower]['freq'] += 1
            else:
                container[collocation_lower] = {
                    'freq': 1, 
                    'raw': colloc_text, 
                    'nnp_rate': proper_noun_rate, 
                    'pos_seq': collocation['pos']
                }

def extract_proper_nouns(sent_toks, container):

    '''
        NNP: Proper Noun Singular
        NNPS: Proper Noun plural
        CC: Coordinating conjunction
        IN: Preposition or subordinating conjunction

        The goal here is to extract multi-word proper nouns. E.g., 
            "Hurricane harvey" (NNP NNP)
            "Centers for Disease Control" (NNP IN NNP NNP)
            "Federal Emergency Management Agency" (NNP NNP NNP NNP)
        This is achieved extracting a contiguous list of NNP or a contiguous list of NNP interleaved with CC or IN
    '''

    if( len(sent_toks) == 0 ):
        return []

    last_pos = ''
    proper_nouns = {'toks': [[]], 'pos': [[]]}

    for i in range( len(sent_toks) ):

        pos = sent_toks[i]['pos']
        tok = sent_toks[i]['tok']
        if( pos.startswith('NNP') ):
            #label 0
            #match e.g., NNP
            proper_nouns['toks'][-1].append(tok)
            proper_nouns['pos'][-1].append(pos)

        elif( len(proper_nouns['toks'][-1]) != 0 and pos in ['IN', 'CC'] ):
            #label 1
            #match e.g., NNP IN or NNP CC
            proper_nouns['toks'][-1].append(tok)
            proper_nouns['pos'][-1].append(pos)

        elif( len(proper_nouns['toks'][-1]) != 0 ):
            
            #label 2
            if( len(proper_nouns['toks'][-1]) == 1 ):
                #label 2.0
                #match NNP (single NNP, violates multi-word condition)
                proper_nouns['toks'][-1] = []
                proper_nouns['pos'][-1] = []
            
            elif( last_pos not in ['NNP', 'NNPS'] ):
                #label 2.1
                #match e.g., NNP IN or NNP CC or NNP NNP IN or NNP IN NNP CC
                proper_nouns['toks'][-1] = proper_nouns['toks'][-1][:-1]
                proper_nouns['pos'][-1] = proper_nouns['pos'][-1][:-1]
                
                if( len(proper_nouns['toks'][-1]) == 1 ):
                    #label 2.2
                    #match e.g, originally NNP CC, here means CC removed leaving just single NNP - violates multi-word condition
                    proper_nouns['toks'][-1] = []
                    proper_nouns['pos'][-1] = []
                else:
                    #label 2.3
                    #match e.g, originally NNP NNP IN, here means IN was remove leaving NNP NNP
                    proper_nouns['toks'].append( [] )
                    proper_nouns['pos'].append( [] )
            
            else:
                #label 3
                #match e.g, NNP NNP IN NNP, here means multi-word proper noun group has ended so create empty slots for next proper noun group
                proper_nouns['toks'].append( [] )
                proper_nouns['pos'].append( [] )

        last_pos = pos


    if( len(proper_nouns['toks'][0]) == 0 ):
        #here means no proper noun was found
        return []
    else:
        
        if( len(proper_nouns['toks'][-1]) == 0 ):
            #here means at least one multi-word proper noun was found, so remove last empty slot
            proper_nouns['toks'] = proper_nouns['toks'][:-1]
            proper_nouns['pos'] = proper_nouns['pos'][:-1]

    for i in range( len(proper_nouns['toks']) ):

        proper_noun = proper_nouns['toks'][i]
        proper_noun = ' '.join(proper_noun)
        proper_noun_lower = proper_noun.lower()
        
        proper_noun_rate = round(proper_nouns['pos'][i].count('NNP')/len(proper_nouns['pos'][i]), 4)
        
        if( proper_noun_lower in container ):
            container[proper_noun_lower]['freq'] += 1
        else:
            container[proper_noun_lower] = {
                'freq': 1, 
                'raw': proper_noun, 
                'nnp_rate': proper_noun_rate, 
                'pos_seq': proper_nouns['pos'][i]
            }

    return proper_nouns

def rank_proper_nouns(multi_word_proper_nouns):
    multi_word_proper_nouns = sorted(multi_word_proper_nouns.items(), key=lambda prnoun: prnoun[1]['freq']*prnoun[1]['nnp_rate'], reverse=True)
    
    for i in range( len(multi_word_proper_nouns) ):
        multi_word_proper_nouns[i][1]['rank'] = i

    return multi_word_proper_nouns

def get_ngram_pos(toks, key_indx):

    '''
        Given ngram: 'convention center'
        
            found at index: 13 
            in this sentence: at the brown convention center
            sentence toks: ['at', the', 'brown', 'convention', 'center']

        The purpose of this function is to return sentence toks index: 3
    '''
    offset = 0
    for i in range(len(toks)):

        if( offset == key_indx ):
            return i

        offset += len(toks[i]) + 1

    return -1

def indx_where_ngram_ends(st_indx, ngram_toks, sent_toks):

    '''

        Case 1: where sent_toks contains stopwords making it hard to match ngram that already has stopwords removed:
            Given st_index = 2 (sent_toks index where ngram_toks starts)
            Given ngram_toks: ['orange', 'new', 'black']
            Given sent_toks: ['best', 'is', 'orange', 'is', 'the', 'new', 'black']

            The goal of this function is return 5: length of substring (['orange', 'is', 'the', 'new', 'black']) with stopwords that encompass ngram_toks

        Negative Case 1:
            Given st_index = 5 (sent_toks index where ngram_toks starts)
            Given ngram_toks: ['convention', 'center']
            Given sent_toks: ['it', 'is', 'the', 'brown', 'r.', 'convention', 'center']

            The goal of this function is return 2: length of substring ([convention', 'center']) without stopwords that encompass ngram_toks
    '''
    
    j = st_indx
    length = 0
    start = -1
    match_count = 0

    for i in range(len(ngram_toks)):

        while j < len(sent_toks):
            
            length += 1
            
            ngram_tok = ngram_toks[i].strip().lower()
            sent_tok = sent_toks[j].strip().lower()
            
            #to avoid situations where sent_tok is has some additional characters (due to different formatting from ngram)
            #e.g., given ngram_tok "aransas", this should match sent_tok: "aransas'"
            if( sent_tok.find(ngram_tok) != -1 ):
                
                if( start == -1 ):
                    start = j

                j += 1
                match_count += 1

                break

            j += 1

    if( match_count == len(ngram_toks) ):
        #all ngram_toks where found
        return start, length
    else:
        return start, -1

def is_ngram_subset(parent, child, stopwords):

    parent = parent.strip().lower()
    child = child.strip().lower()

    #perfect match
    if( parent.find(child) != -1 ):
        return True

    parent = rmStopwords(parent, stopwords)
    child = rmStopwords(child, stopwords)

    #match when stopwords
    if( ' '.join(parent).find(' '.join(child)) != -1 ):
        return True

    ov = overlapFor2Sets( set(parent), set(child) )
    if( ov != 1 ):
        return False

    return isMatchInOrder(child, parent)

def get_sentence_match_ngram(ngram, ngram_toks, sentences, doc_indx, doc_id):

    phrase_cands = []

    for i in range(len(sentences)):
        
        ori_sent = sentences[i]['sentence'].replace('\n', ' ')
        sentence = ori_sent.lower()
        indx = sentence.find(ngram)

        if( indx != -1 ):

            sentence = sentence.strip()
            sent_toks = phraseTokenizer(sentence)
            
            ngram_start_indx = get_ngram_pos(sent_toks, indx)
            ngram_start, ngram_length = indx_where_ngram_ends(ngram_start_indx, ngram_toks, sent_toks)

            if( ngram_start_indx == -1 ):
                logger.debug('\nDID NOT FIND NGRAM POST SPLITTING' * 10)
                logger.debug('\tngram: ' + str(ngram))
                logger.debug('\tsentence: ' + sentence)
                logger.debug('\tsent_tok not printed')
                logger.debug('')
                continue

            sentence_dets = {
                'ori_sent': ori_sent,
                'sent_indx': i,
                'doc_indx': doc_indx,
                'doc_id': doc_id,
                'toks': sent_toks,
                'ngram_start_indx': ngram_start_indx,
                'ngram_length': ngram_length
            }

            phrase_cands.append(sentence_dets)


    return phrase_cands

def rank_mltwd_proper_nouns(ngram, ngram_toks, sentences, params=None):
    
    if( params is None ):
        params = {}
    
    params.setdefault('mvg_window_min_proper_noun_rate', 0.5)
    params.setdefault('max_ngram', 10)
    params.setdefault('base_ngram', 2)
    sent_count = len(sentences)
    ngram = ngram.strip()

    if( sent_count == 1 or ngram == '' ):
        #it's possible for ngram = '', search for 'sumgram_history'
        return {}

    window_size = 1
    max_sent_toks = 0
    final_multi_word_proper_noun = {}
    max_multiprpnoun_lrb = {}
    
    max_window_size = (params['max_ngram'] - params['base_ngram'])/2
    logger.debug( '\tmax_window_size: ' + str(max_window_size) )
    logger.debug( '\tbase_ngram: ' + str(params['base_ngram']) )

    while window_size <= max_window_size:

        max_multiprpnoun_lrb[window_size] = {'lrb': '', 'ngram': '', 'rate': 0}
        phrase_counts = { 'left': {}, 'right': {}, 'both': {} }
        proper_noun_phrases = {'left': '', 'right': '', 'both': ''}
        
        for i in range(sent_count):
            
            sent = sentences[i]
            
            ngram_start = sent['ngram_start_indx']
            ngram_length = sent['ngram_length']
            if( ngram_start == -1 or ngram_length == -1 ):
                continue

            sent_toks_count = len(sent['toks'])

            logger.debug( '\n\tbase ngram: ' + str(ngram_toks) )
            logger.debug( '\tngram in sent (start/length): ' + str(ngram_start) + '/' + str(ngram_length) )
            logger.debug( '\tsent keys: ' + str(sent.keys()) )
            logger.debug( '\tori: ' + sent['ori_sent'] )
            logger.debug( '\tsent: ' + str(i) + ' of ' + str(sent_count) + ': ' + str(sent['toks']) )
            logger.debug( '\tsent_len: ' + str(sent_toks_count))

            if( window_size == 1 and sent_toks_count > max_sent_toks ):
                max_sent_toks = sent_toks_count

            ngram_prefix = sent['toks'][ngram_start - window_size:ngram_start]
            ngram_suffix = sent['toks'][ngram_start + ngram_length:ngram_start + ngram_length + window_size]
            
            ngram_prefix = ' '.join(filter(None, ngram_prefix)).strip()
            ngram_suffix = ' '.join(filter(None, ngram_suffix)).strip()
            
            proper_noun_phrases['left'] = ngram_prefix + ' ' + ngram
            proper_noun_phrases['right'] = ngram + ' ' + ngram_suffix
            proper_noun_phrases['both'] = ngram_prefix + ' ' + ngram + ' ' + ngram_suffix
            
            for lrb in ['left', 'right', 'both']:

                '''
                    #does not account for ties consider: if( max_multiprpnoun_lrb[window_size-1]['lrb'] != lrb and max_multiprpnoun_lrb[window_size-1]['tie'] == False ):
                    if( window_size > 1 ):
                        if( max_multiprpnoun_lrb[window_size-1]['lrb'] != lrb ):
                            
                            if( params['debug_verbose'] ):
                                print('\t\tskipping:', lrb, 'since previous lrb was below threshold')
                            continue
                '''

                multi_word_proper_noun_lrb = proper_noun_phrases[lrb].strip()

                if( multi_word_proper_noun_lrb != ngram ):
                    
                    phrase_counts[ lrb ].setdefault( multi_word_proper_noun_lrb, {'freq': 0, 'rate': 0} )
                    phrase_counts[ lrb ][multi_word_proper_noun_lrb]['freq'] += 1
                    phrase_counts[ lrb ][multi_word_proper_noun_lrb]['rate'] = round( phrase_counts[lrb][multi_word_proper_noun_lrb]['freq']/sent_count, 4 )
                    phrase_counts[ lrb ][multi_word_proper_noun_lrb]['details'] = {
                        'ngram': multi_word_proper_noun_lrb,
                        'sent_freq': sent_count,
                        'ngram_freq': phrase_counts[ lrb ][multi_word_proper_noun_lrb]['freq'],
                        'window_size': window_size
                    }

                    logger.debug( '\t\t' + lrb + ': ' + multi_word_proper_noun_lrb + ', freq: ' + str(phrase_counts[ lrb ][multi_word_proper_noun_lrb]['freq']) )
            
            logger.debug( '\t\tsent_count: ' + str(sent_count) )
            logger.debug( '\t\twindow_size: ' + str(window_size) )
        logger.debug('\n\twindow_size: ' + str(window_size) + ' results:')


        #find multi-word proper noun with the highest frequency for left, right, and both sentence building policies
        multiprpnoun_lrb = sorted(phrase_counts['both'].items(), key=lambda x: x[1]['rate'], reverse=True)
        
        if( len(multiprpnoun_lrb) != 0 ):
            
            rate = multiprpnoun_lrb[0][1]['rate']
            if( rate >= params['mvg_window_min_proper_noun_rate'] ):
                #preference is given to good quality both ngram, if only poor quality both is available, then check left and right
                logger.debug( '\t\tmax both: ' + str(multiprpnoun_lrb[0]) )
                logger.debug( '\t\tboth (longer) is preference will disregard left & right')

                max_multiprpnoun_lrb[window_size]['ngram'] = multiprpnoun_lrb[0][0]
                max_multiprpnoun_lrb[window_size]['rate'] = rate
                max_multiprpnoun_lrb[window_size]['lrb'] = 'both'
                max_multiprpnoun_lrb[window_size]['details'] = multiprpnoun_lrb[0][1]['details']
            else:
                logger.debug( '\t\tboth rate < mvg_window_min_proper_noun_rate: ' + str(multiprpnoun_lrb[0]) )


        if( max_multiprpnoun_lrb[window_size]['rate'] == 0 ):
            for lrb in ['left', 'right']:
                
                multiprpnoun_lrb = sorted(phrase_counts[lrb].items(), key=lambda x: x[1]['rate'], reverse=True)
                
                if( len(multiprpnoun_lrb) != 0 ):
                    
                    logger.debug('\t\tmax ' + lrb + ': ' + str(multiprpnoun_lrb[0]))
                    
                    rate = multiprpnoun_lrb[0][1]['rate']
                    if( rate > max_multiprpnoun_lrb[window_size]['rate'] ):
                        max_multiprpnoun_lrb[window_size]['ngram'] = multiprpnoun_lrb[0][0]
                        max_multiprpnoun_lrb[window_size]['rate'] = rate
                        max_multiprpnoun_lrb[window_size]['lrb'] = lrb
                        max_multiprpnoun_lrb[window_size]['details'] = multiprpnoun_lrb[0][1]['details']

        logger.debug('\tlast max for this window_size: ' + str(max_multiprpnoun_lrb[window_size]))
        logger.debug('\tmax_sent_toks: ' + str(max_sent_toks))
        logger.debug('')

        if( params['mvg_window_min_proper_noun_rate'] > max_multiprpnoun_lrb[window_size]['rate'] or window_size == max_sent_toks ):
            logger.debug("\tbreaking criteria reached: mvg_window_min_proper_noun_rate > max_multiprpnoun_lrb[window_size]['rate'] OR window_size (" + str(window_size) + ") == max_sent_toks (" + str(max_sent_toks) + ")")
            logger.debug('\tmvg_window_min_proper_noun_rate: ' + str(params['mvg_window_min_proper_noun_rate']))
            logger.debug("\tmax_multiprpnoun_lrb[window_size]['rate']: " + str(max_multiprpnoun_lrb[window_size]['rate']))
            break

        window_size += 1


    logger.debug('\n\tfinal winning candidates: ')
    ngram_length_window_size_map = {}
    for window in max_multiprpnoun_lrb:
    
        logger.debug( '\t\tfinal winning candidate ' + str(window) + ': ' + str(max_multiprpnoun_lrb[window]) )
        ngram_length_window_size_map[window] = len(max_multiprpnoun_lrb[window]['ngram'].split(' '))

    #0: window size, 1: ngram length, give preference to longer ngrams
    ngram_length_window_size_map = sorted( ngram_length_window_size_map.items(), key=lambda x: x[1], reverse=True )
    logger.debug('')



    for window_size in ngram_length_window_size_map:
        
        #0: window size, 1: ngram length, give preference to longer ngrams
        window_size = window_size[0]
        #get best match longest multi-word ngram 
        if( max_multiprpnoun_lrb[window_size]['rate'] >= params['mvg_window_min_proper_noun_rate'] ):
            
            final_multi_word_proper_noun['proper_noun'] = max_multiprpnoun_lrb[window_size]['ngram']
            final_multi_word_proper_noun['rate'] = max_multiprpnoun_lrb[window_size]['rate']
            final_multi_word_proper_noun['details'] = max_multiprpnoun_lrb[window_size]['details']
            
            logger.debug('\tfinal winning max: ' + str(max_multiprpnoun_lrb[window_size]))
            logger.debug('\twindow_size: ' + str(window_size))
            break
            

    return final_multi_word_proper_noun

def pos_glue_split_ngrams(top_ngrams, k, pos_glue_split_ngrams_coeff, ranked_multi_word_proper_nouns, params):

    stopwords = get_dual_stopwords( params['add_stopwords'] )
    multi_word_proper_noun_dedup_set = set()#it's possible for different ngrams to resolve to the same multi-word proper noun so deduplicate favoring higher ranked top_ngrams
    for i in range( len(top_ngrams) ):
        
        if( i == k - 1 ):
            break

        ngram = top_ngrams[i]['ngram']
        for mult_wd_prpnoun in ranked_multi_word_proper_nouns:
            
            multi_word_proper_noun = mult_wd_prpnoun[0]
            match_flag = is_ngram_subset(multi_word_proper_noun, ngram, stopwords)
            
            if( match_flag ):

                if( ngram == multi_word_proper_noun or mult_wd_prpnoun[1]['freq'] < top_ngrams[i]['term_freq'] * pos_glue_split_ngrams_coeff ):
                    #this ngram exactly matched a multi_word_proper_noun, and thus very unlikely to be a fragment ngram to be replaced

                    #to avoid replacing high-quality ngram with poor-quality ngram - start
                    '''
                        rationale for: mult_wd_prpnoun[1]['freq'] < top_ngrams[i]['term_freq']/2
                        bad replacement:
                            ngram: tropical storm (freq: 121)
                            replaced with multi_word_proper_noun: ddhhmm tropical storm harvey discussion number
                            rank: {'freq': 5, 'raw': 'DDHHMM TROPICAL STORM HARVEY DISCUSSION NUMBER', 'nnp_rate': 1.0, 'rank': 358}
            
                            ngram: tropical cyclone (43)
                            replace with multi_word_proper_noun: wikipedia tropical cyclone
                            rank: {'freq': 1, 'raw': 'WIKIPEDIA TROPICAL CYCLONE', 'nnp_rate': 1.0, 'rank': 3340}

                        good replacement:
                            ngram: national hurricane (67)
                            match: national hurricane center
                            rank: {'freq': 68, 'raw': 'National Hurricane Center', 'nnp_rate': 1.0, 'rank': 11}

                            ngram: gulf mexico (56)
                            match: gulf of mexico
                            rank: {'freq': 46, 'raw': 'Gulf of Mexico', 'nnp_rate': 0.6667, 'rank': 41}
                    '''
                    #to avoid replacing high-quality ngram with poor-quality ngram - end
                    pass
                else:
                    
                    new_ngram_dct = {
                        'prev_ngram': top_ngrams[i]['ngram'],
                        'annotator': 'pos',
                        'cur_freq': mult_wd_prpnoun[1]['freq'],
                        'cur_pos_sequence': mult_wd_prpnoun[1]['pos_seq']
                    }

                    if( multi_word_proper_noun in multi_word_proper_noun_dedup_set ):
                        top_ngrams[i]['ngram'] = ''
                        new_ngram_dct['cur_ngram'] = ''
                    else:
                        top_ngrams[i]['ngram'] = multi_word_proper_noun
                        new_ngram_dct['cur_ngram'] = multi_word_proper_noun
                        
                        multi_word_proper_noun_dedup_set.add(multi_word_proper_noun)

                    top_ngrams[i].setdefault('sumgram_history', [])
                    top_ngrams[i]['sumgram_history'].append(new_ngram_dct)
                
                break

def mvg_window_glue_split_ngrams(top_ngrams, k, all_doc_sentences, params=None):

    logger.debug('\nmvg_window_glue_split_ngrams():')
    if( params is None ):
        params = {}

    multi_word_proper_noun_dedup_set = set()#it's possible for different ngrams to resolve to the same multi-word proper noun so deduplicate favoring higher ranked top_ngrams

    for i in range( len(top_ngrams) ):
        
        if( i == k - 1 ):
            break
        
        ngram = top_ngrams[i]['ngram']
        if( ngram == '' ):
            #ngram could be blank due to rm by pos_glue_split_ngrams
            continue
        
        ngram_toks = phraseTokenizer(ngram)#processed in a similar method as the sentences in get_sentence_match_ngram

        phrase_cands = []
        phrase_cands_minus_toks = []

        logger.debug('\t' + str(i) + ' ngram: ' + str(ngram))

        for doc_dct in top_ngrams[i]['postings']:
            
            doc_indx = doc_dct['doc_indx']
            doc_id = doc_dct['doc_id']
            phrase_cands += get_sentence_match_ngram( ngram, ngram_toks, all_doc_sentences[doc_indx], doc_indx, doc_id )

        for phrase in phrase_cands:
            phrase_cands_minus_toks.append({
                'sentence': phrase['ori_sent'],
                'sent_indx': phrase['sent_indx'],
                'doc_indx': phrase['doc_indx'],
                'doc_id': phrase['doc_id']
            })
            
        multi_word_proper_noun = rank_mltwd_proper_nouns( ngram, ngram_toks, phrase_cands, params=params )
        if( len(multi_word_proper_noun) != 0 ):

            new_ngram_dct = {
                'prev_ngram': top_ngrams[i]['ngram'],
                'annotator': 'mvg_window'
            }
            
            if( multi_word_proper_noun['proper_noun'] in multi_word_proper_noun_dedup_set ):
                top_ngrams[i]['ngram'] = ''
                new_ngram_dct['cur_ngram'] = ''
            else:
                top_ngrams[i]['ngram'] = multi_word_proper_noun['proper_noun']
                
                new_ngram_dct['cur_ngram'] = multi_word_proper_noun['proper_noun']
                new_ngram_dct['mvg_window_ngram'] = multi_word_proper_noun['details']

                multi_word_proper_noun_dedup_set.add( multi_word_proper_noun['proper_noun'] )

            top_ngrams[i].setdefault('sumgram_history', [])
            top_ngrams[i]['sumgram_history'].append(new_ngram_dct)

        top_ngrams[i]['parent_sentences'] = phrase_cands_minus_toks
            
        logger.debug('*' * 200)
        logger.debug('*' * 200)
        logger.debug('')



def rm_empty_and_stopword_ngrams(top_ngrams, k, stopwords):

    final_top_ngrams = []

    for i in range( len(top_ngrams) ):
        
        if( i == k - 1 ):
            break 

        if( top_ngrams[i]['ngram'] == '' ):
            continue

        #check if top_ngrams[i]['ngram'] has stopword, if so skip - start
        match_flag = False
        for stpwrd in stopwords:
            
            match_flag = is_ngram_subset(parent=top_ngrams[i]['ngram'], child=stpwrd, stopwords={})
            if( match_flag ):
                break

        if( match_flag is True ):
            continue

        #check if top_ngrams[i]['ngram'] has stopword, if so skip - end

        final_top_ngrams.append( top_ngrams[i] )

    return final_top_ngrams

def rm_subset_top_ngrams(top_ngrams, k, rm_subset_top_ngrams_coeff, params):
    
    ngram_tok_sizes = {}
    stopwords = get_dual_stopwords( params['add_stopwords'] )

    for i in range( len(top_ngrams) ):
        
        if( i == k - 1 ):
            break

        ngram = top_ngrams[i]['ngram']
        top_ngrams[i]['adopted_child'] = False
        ngram_tok_sizes[i] = len( phraseTokenizer(ngram) )
    
    #prioritize longer top_ngrams
    ngram_tok_sizes = sorted(ngram_tok_sizes.items(), key=lambda x: x[1], reverse=True)
    
    for ngram_indx in ngram_tok_sizes:        
        
        parent_indx = ngram_indx[0]
        parent_ngram_cand = top_ngrams[parent_indx]['ngram']

        if( parent_ngram_cand == '' ):
            continue


        for child_indx in range( len(ngram_tok_sizes) ):
            
            if( parent_indx == child_indx ):
                continue

            child_ngram_cand = top_ngrams[child_indx]['ngram']
            if( child_ngram_cand == '' ):
                continue
            
            if( is_ngram_subset(parent_ngram_cand, child_ngram_cand, stopwords) ):

                if( parent_indx < child_indx ):
                    #if parent is at a higher rank (lower index) than child, so delete child, parent remains unmodified (top_ngrams[parent_indx]['ngram'] already has parent_ngram_cand)
                    top_ngrams[child_indx]['ngram'] = ''
                else:
                    #parent (longer) is at a lower rank (higher index) than child, 
                    #INSTEAD OF: delete parent to give preference shorter highly ranked child, child remains unmodified
                    #replace child (higher rank, shorter ngram) with parent (lower rank, longer ngram) if parent's TF >= child's TF * 1/k,
                    #multiple children may fulfil this criteria, so parent should adopt (replace) the first child and remove subsequent children that fulfill this criteria

                    top_ngrams[parent_indx]['ngram'] = ''
                    if( top_ngrams[parent_indx]['term_freq'] >= top_ngrams[child_indx]['term_freq'] * rm_subset_top_ngrams_coeff ):

                        if( top_ngrams[parent_indx]['adopted_child'] == False ):
                            
                            top_ngrams[parent_indx]['adopted_child'] = True
                            top_ngrams[child_indx]['ngram'] = parent_ngram_cand

                            new_ngram_dct = {
                                'prev_ngram': child_ngram_cand,
                                'cur_ngram': parent_ngram_cand,
                                'cur_freq': top_ngrams[parent_indx]['term_freq'],
                                'annotator': 'subset'
                            }

                            if( 'sumgram_history' in top_ngrams[parent_indx] ):
                                #this parent has a history
                                new_ngram_dct['cur_ngram_sumgram_history'] = top_ngrams[parent_indx]['sumgram_history']

                            top_ngrams[child_indx].setdefault('sumgram_history', [])
                            top_ngrams[child_indx]['sumgram_history'].append(new_ngram_dct)

                            logger.debug('\teven though parent (' + str(parent_indx) + ') is in lower index than child: ' + str(child_indx))
                            logger.debug('\treplacing child_ngram_cand:' + '"' + child_ngram_cand + '" with parent_ngram_cand: "' + parent_ngram_cand + '"')
                            logger.debug('\treplacing parent/child tf: ' + str(top_ngrams[parent_indx]['term_freq']) + ' ' + str(top_ngrams[child_indx]['term_freq']) )
                            logger.debug('')
                        else:
                            
                            top_ngrams[child_indx]['ngram'] = ''

                            logger.debug('\teven though parent (' + str(parent_indx) + ') is in lower index than child: ' + str(child_indx))
                            logger.debug('\twould have replaced child_ngram_cand: "' + child_ngram_cand + '" with parent_ngram_cand: "' + parent_ngram_cand + '"')
                            logger.debug('\twould have replaced parent/child tf:' + str(top_ngrams[parent_indx]['term_freq']) + '/' + str(top_ngrams[child_indx]['term_freq']))
                            logger.debug('\tbut this parent has already adopted a child so delete this child.')
                            logger.debug('')
                    

    return top_ngrams
    
def print_top_ngrams(n, top_ngrams, top_sumgram_count, params=None):

    if( params is None ):
        params = {}

    if( len(top_ngrams) == 0 ):
        return

    params.setdefault('ngram_printing_mw', 50)
    params.setdefault('title', '')
    doc_len = params.get('doc_len', None)
    default_color = '49m'
    tf_or_df = ''

    if( 'last_ngram' in params['state'] ):
        last_ngram = params['state']['last_ngram']
    else:
        last_ngram = {}

    mw = params['ngram_printing_mw']
    ngram_count = len(top_ngrams)

    print('\nSummary for {} top sumgrams (base n: {}, docs: {:,}):'.format(ngram_count, n, doc_len))
    if( params['title'] != '' ):
        print( params['title'])


    if( params['binary_tf_flag'] is True ):
        tf_or_df = 'DF'
    else:
        tf_or_df = 'TF'


    if( params['base_ngram_ansi_color'] == '' ):
        print( '{:^6} {:^6} {:<7} {:<30} {:<{mw}}'.format('Rank', tf_or_df, tf_or_df + '-Rate', 'Base ngram', 'Sumgram', mw=mw))
    else:
        print( '{:^6} {:^6} {:<7} {:<30} {:<{mw}}'.format('Rank', tf_or_df, tf_or_df + '-Rate', 'Base ngram', getColorTxt('Sumgram', default_color), mw=mw))

    for i in range(top_sumgram_count):
        
        if( i == ngram_count ):
            break

        
        ngram = top_ngrams[i]
        ngram_txt = ngram['ngram']
        if( len(ngram_txt) > mw ) :
            ngram_txt = ngram_txt[:mw-15] + '...'


        base_ngram = ngram_txt
        if( 'sumgram_history' in ngram ):
            
            base_ngram = ngram['sumgram_history'][0]['prev_ngram']
            
            if( params['base_ngram_ansi_color'] != '' ):
                
                base_ngram_color = getColorTxt(base_ngram, params['base_ngram_ansi_color'])
                prev_ngram_txt = ngram_txt
                ngram_txt = re.sub(base_ngram, base_ngram_color, ngram_txt)
                
                if( prev_ngram_txt == ngram_txt ):
                    #substitution did not happen, so use default color
                    ngram_txt = getColorTxt( ngram_txt, default_color )
        
        elif( params['base_ngram_ansi_color'] != '' ):
            ngram_txt = getColorTxt(ngram_txt, default_color)

        print( "{:^6} {:^6} {:^7} {:<30} {:<{mw}}".format(i+1, ngram['term_freq'], "{:.2f}".format(ngram['term_rate']), base_ngram, ngram_txt, mw=mw))

    if( len(last_ngram) != 0 ):
        if( params['min_df'] != 1 ):
            print( 'last ngram with min_df = ' + str(params['min_df']) + ' (index/' + tf_or_df + '/' + tf_or_df + '-Rate): ' + last_ngram['ngram'] + ' (' + str(last_ngram['rank'])  + '/' + str(last_ngram['term_freq']) + '/' + str(last_ngram['term_rate']) + ')')

def print_top_doc_sent(report):

    if( 'ranked_docs' in report ):
        if( len(report['ranked_docs']) != 0 ):
            logger.info('\nTop ranked document index: ' + str(report['ranked_docs'][0]['doc_id']))


    if( 'ranked_sentences' in report ):
        if( len(report['ranked_sentences']) != 0 ):
            logger.info('\nTop ranked sentence: ' + str(report['ranked_sentences'][0]['sentence']) )

def extract_top_ngrams(doc_lst, doc_dct_lst, n, params):

    logger.debug('\nextract_top_ngrams(): token_pattern: ' + params['token_pattern'])
    
    '''
        Note on unified all_doc_sentences and top_ngrams text processing:
        #By default, all_doc_sentences are generated from stanford corenlp sentence annotator. The else: case was an attempt to similarly process the text used to generate top_ngrams in the same fashion (
        utilizing corenlp's context-sensitive tokenizer (e.g, dot in aug. belongs to month, and not to be used as sentence boundary)). But requires feeding CountVectorizer the vocabulary (ngrams), which
        is easy to do unless when unigrams are to be generated. I didn't consider it worth the effort to custom build my n-gram generator, so I opted to not proceed to treat top_ngrams as all_doc_sentences.
        The consequence of this is that some ngrams from top_ngrams may not match those extracted from all_sentences.
        
        from genericCommon import nlpBuildVocab
        vocab = nlpBuildVocab(doc_lst)
        if( len(vocab) == 0 ):
            count_vectorizer = CountVectorizer(stop_words=getStopwordsSet(), token_pattern=params['token_pattern'], ngram_range=(n, n), binary=params['binary_tf_flag'])
        else:
            count_vectorizer = CountVectorizer(stop_words=None, vocabulary=vocab, token_pattern=params['token_pattern'], ngram_range=(n, n), binary=params['binary_tf_flag'])
    '''
    
    doc_count = len(doc_dct_lst)
    if( doc_count == 1 ):
        binary_tf_flag = False
    else:
        binary_tf_flag = True

    bif_stopwords = bifurcate_stopwords( params['add_stopwords'] )
    stopwords = bif_stopwords['unigrams'] | params['add_stopwords_file'] if params['no_default_stopwords'] is True else list(getStopwordsSet() | bif_stopwords['unigrams'] | params['add_stopwords_file'])
    min_df = params['min_df']    

    try:
        if( isinstance(min_df, str) ):
            if( min_df.find('.') == -1 ):
                min_df = int(min_df)
            else:
                min_df = float(min_df)
    except:
        min_df = 1

    
    params['min_df'] = min_df
    count_vectorizer = CountVectorizer(stop_words=stopwords, token_pattern=params['token_pattern'], ngram_range=(n, n), binary=binary_tf_flag, min_df=min_df)
    
    logger.debug('\tfit transfrom - start')
    try:
        #tf_matrix is a binary TF matrix if doc_lst.len > 1, non-binary otherwise
        tf_matrix = count_vectorizer.fit_transform(doc_lst).toarray()

        all_col_sums_tf = np.sum(tf_matrix, axis=0)
        all_non_zero = np.argwhere(tf_matrix != 0)
    except:
        genericErrorInfo()
        return []
    logger.debug('\tfit transfrom - end')
    
    #every entry in list top_ngrams is of type: (a, b), a: term, b: term position in TF matrix
    try:
        top_ngrams = count_vectorizer.get_feature_names_out()
    except AttributeError:
        top_ngrams = count_vectorizer.get_feature_names()
    except:
        genericErrorInfo()
        return []

    filtered_top_ngrams = {}
    total_freq = 0
    
    logger.debug('\ttf_matrix shape: ' + str(tf_matrix.shape))

    for doc_indx, i in all_non_zero:
        #i is index of vocab (top_ngrams[i])
        if( top_ngrams[i] in bif_stopwords['multigrams'] ):
            continue
        
        #find a simpler way to convert Int64 to native int
        doc_indx = int(doc_indx)
        
        filtered_top_ngrams.setdefault(i, get_ngram_dct(top_ngrams[i], -1, []))

        if( filtered_top_ngrams[i]['term_freq'] == -1 ):
            
            if( binary_tf_flag is True ):
                col_sum_tf = int(all_col_sums_tf[i])
            else:
                matrix_col = tf_matrix[:,i]
                col_sum_tf = int(matrix_col[0])

            filtered_top_ngrams[i]['term_freq'] = col_sum_tf
            total_freq += col_sum_tf

        filtered_top_ngrams[i]['postings'].append({
            'doc_indx': doc_indx, #Int64 to native int
            'doc_id': doc_dct_lst[doc_indx]['doc_id'],
            'doc_details': fmt_posting( doc_dct_lst[doc_indx] )
        })


    if( doc_count == 1 ):
        N = total_freq
        params['tf_label'] = 'Term Frequency'
    else:
        N = doc_count
        params['tf_label'] = 'Document Frequency'

    params['tf_normalizing_divisor'] = N
    params['collection_doc_count'] = doc_count
    filtered_top_ngrams = sorted(filtered_top_ngrams.items(), key=lambda x: x[1]['term_freq'], reverse=True)
    filtered_top_ngrams = [x[1] for x in filtered_top_ngrams]

    for i in range(len(filtered_top_ngrams)):
        filtered_top_ngrams[i]['term_rate'] = filtered_top_ngrams[i]['term_freq']/N

    if( len(filtered_top_ngrams) != 0 ):
        params['state']['last_ngram'] = filtered_top_ngrams[-1]
        params['state']['last_ngram']['rank'] = len(filtered_top_ngrams)

    return filtered_top_ngrams

def update_doc_indx(report, doc_id_new_doc_indx_map):
    
    if( len(doc_id_new_doc_indx_map) == 0 ):
        return

    '''
        After ranking documents, the doc_indx in 'ranked_sentences' and top_sumgrams.postings and top_sumgrams.parent_sentences
        is no longer valid because they reference the initial doc_lst permutation, so fix this with doc_id_new_doc_indx_map
    '''

    #update report['ranked_sentences']
    if( 'ranked_sentences' in report ):
        for i in range( len(report['ranked_sentences']) ):
            
            doc_id = report['ranked_sentences'][i]['doc_id']
            if( doc_id in doc_id_new_doc_indx_map ):
                report['ranked_sentences'][i]['doc_indx'] = doc_id_new_doc_indx_map[doc_id]

    #update report['top_sumgrams'][*]['postings'] and report['top_sumgrams'][*]['parent_sentences']
    for i in range( len(report['top_sumgrams']) ):
        for opt in ['postings', 'parent_sentences']:
            for j in range( len(report['top_sumgrams'][i][opt]) ):

                doc_id = report['top_sumgrams'][i][opt][j]['doc_id']
                if( doc_id in doc_id_new_doc_indx_map ):
                    report['top_sumgrams'][i][opt][j]['doc_indx'] = doc_id_new_doc_indx_map[doc_id]

def get_user_stopwords(add_stopwords):

    all_stopwords = []
    for s in add_stopwords:
        all_stopwords += s['text'].split()

    return all_stopwords

def get_top_sumgrams(doc_dct_lst, n=2, params=None):

    if( params is None or isinstance(params, dict) == False ):
        params = {}
    
    params.setdefault('referrer', '')
    if( params['referrer'] != 'main' ):
        #measure to avoid re-using previous state of params when get_top_sumgrams is called multiple times from a script
        params = copy.deepcopy(params)
    
    report = {}
    if( len(doc_dct_lst) == 0 ):
        return report

    if( n < 1 ):
        n = 1

    params = get_default_args(params)
    params['state'] = {}
    params['doc_len'] = len(doc_dct_lst)
    params['add_stopwords'] = set([ s.strip().lower() for s in params['add_stopwords'] if s.strip() != '' ])
    params['add_stopwords_file'] = set([ s.strip().lower() for s in params['add_stopwords_file'] if s.strip() != '' ])
    params.setdefault('binary_tf_flag', True)#Multiple occurrence of term T in a document counts as 1, TF = total number of times term appears in collection
    nlp_addr = 'http://' + params['corenlp_host'] + ':' + params['corenlp_port']

    
    if( params['sentence_tokenizer'] == 'ssplit' ):
        params['stanford_corenlp_server'] = nlpIsServerOn( addr=nlp_addr )
    else:
        params['stanford_corenlp_server'] = False


    logger.debug('\nget_top_sumgrams():')
    if( params['stanford_corenlp_server'] == False and params['sentence_tokenizer'] == 'ssplit' ):
        
        logger.info('\n\tAttempting to start Stanford CoreNLP Server (we need it to segment sentences)\n')
        
        nlpServerStartStop('start', host=params['corenlp_host'], port=params['corenlp_port'])
        params['stanford_corenlp_server'] = nlpIsServerOn( addr=nlp_addr )
    
    #doc_dct_lst: {doc_id: , text: }
    logger.debug('\tsentence segmentation - start')
    parallel_nlp_add_sents(doc_dct_lst, params)

    #main algorithm step 1 - start
    
    '''
        1. Add plain_text into doc_lst
        2. Create dict of <doc_indx, [doc_sentences]>, sentences can can be segmented by either cornlp ssplit (by parallel_nlp_add_sents()) or regex
        3. Extract multi_word_proper_nouns by using rules (e.g., NNP IN NNP NNP) with extract_proper_nouns()
    '''

    doc_lst = []                                        
    all_doc_sentences = {}                              
    multi_word_proper_nouns = {}
    dedup_set = set()
    for i in range(len(doc_dct_lst)):
        
        doc_dct_lst[i].setdefault('doc_id', i)
        doc_lst.append( doc_dct_lst[i]['text'] )

        #placing sentences inside doc_dct_lst[i] accounted for more runtime overhead
        all_doc_sentences[i] = extract_doc_sentences( 
            doc_dct_lst[i], 
            params['sentence_pattern'], 
            dedup_set, 
            multi_word_proper_nouns, 
            params=params
        )

        del doc_dct_lst[i]['text']
        
        if( 'sentences' in doc_dct_lst[i] ):
            del doc_dct_lst[i]['sentences']
    
    multi_word_proper_nouns = rank_proper_nouns(multi_word_proper_nouns)
    #main algorithm step 1 - end
    

    logger.debug('\tsentence segmentation - end')
    logger.debug('\tshift: ' + str(params['shift']))
    
    #step 2 - start
    '''
        step 1
        1. extract_top_ngrams(): Extract top n-grams from text, top is defined by top DF (for multiple documents) or top TF (for single documents)
        2. pos_glue_split_ngrams(): NER (POS tagger) is active: replace children subset ngrams (e.g., "national hurricane") with superset parent multi-word proper noun (e.g., "national hurricane center") extracted by multi_word_proper_nouns(). Subset means overlap is 1.0 and match order is preserved (e.g., "hurricane national" is NOT subset of "national hurricane center" since even though overlap is 1.0, match is out of order)
        3. mvg_window_glue_split_ngrams(): 
        4. rm_subset_top_ngrams(): 
    '''
    #step 2 - end

    top_ngrams = extract_top_ngrams(doc_lst, doc_dct_lst, n, params)
    if( len(top_ngrams) == 0 ):
        return report
    

    if( params['top_sumgram_count'] < 1 or params['top_sumgram_count'] > len(top_ngrams) ):
        params['top_sumgram_count'] = len(top_ngrams)
    
    '''
        shifting is off by default
        shifting is done in an attempt to perform ngram summary on non-top terms 
        this may be required for comparing two different collections that have similar top terms
        so shifting is an attempt to perform process non-top terms in order to find distinguishing ngrams below the top ngrams
    '''
    shift_factor = params['shift'] * params['top_sumgram_count']    
    if( shift_factor >= len(top_ngrams) ):
        shift_factor = 0

    logger.debug('\ttop_ngrams.len: ' + str(len(top_ngrams)))
    if( shift_factor > 0 ):
        
        params['top_ngram_shift_factor'] = shift_factor
        top_ngrams = top_ngrams[shift_factor:]
        logger.debug('\ttop_ngrams.post shift len: ' + str(len(top_ngrams)))


    report = { 'base_ngram': n, 'top_sumgram_count': params['top_sumgram_count']}

    if( params['print_details'] is True ):
        print('doc_lst.len: ' + str(len(doc_dct_lst)))
        print('top ngrams before finding multi-word proper nouns:')
        print_top_ngrams( n, top_ngrams, params['top_sumgram_count'], params=params )
    

    if( params['no_pos_glue_split_ngrams'] == False ):
        pos_glue_split_ngrams( top_ngrams, params['top_sumgram_count'] * 2, params['pos_glue_split_ngrams_coeff'], multi_word_proper_nouns, params )

    #subset top_ngrams will be replace with their supersets, thus shrinking top_sumgram_count counts after this operation, so maximize the chances of reporting user-supplied c, begin by processing: top_sumgram_count * 2
    if( params['no_mvg_window_glue_split_ngrams'] == False ):
        mvg_window_glue_split_ngrams( top_ngrams, params['top_sumgram_count'] * 2, all_doc_sentences, params=params )
    
    
    if( params['print_details'] is True ):
        print('\ntop ngrams after finding multi-word proper nouns:')
        print_top_ngrams( n, top_ngrams, params['top_sumgram_count'], params=params )
    
    
    top_ngrams = rm_subset_top_ngrams( top_ngrams, params['top_sumgram_count'] * 2, params['rm_subset_top_ngrams_coeff'], params )
    if( params['print_details'] is True ):
        print('\ntop ngrams after removing subset phrases:')
        print_top_ngrams( n, top_ngrams, params['top_sumgram_count'], params=params )

    
    top_ngrams = rm_empty_and_stopword_ngrams( top_ngrams, params['top_sumgram_count'] * 2, params['add_stopwords'] )

    doc_id_new_doc_indx_map = {}
    if( params['no_rank_docs'] == False ):
        report['ranked_docs'], doc_id_new_doc_indx_map = get_ranked_docs( top_ngrams, doc_dct_lst )

        if( params['sentences_rank_count'] > 0 and params['no_rank_sentences'] == False ):
            ngram_sentences = combine_ngrams( top_ngrams[:params['top_sumgram_count']] )
            report['ranked_sentences'] = rank_sents_frm_top_ranked_docs( ngram_sentences, report['ranked_docs'], all_doc_sentences, params )
        
        #remove doc_indx
        report['ranked_docs'] = [d[1] for d in report['ranked_docs']]
    
    if( params['print_details'] is True ):
        print('\nfinal sumgrams:')
        print_top_ngrams( n, top_ngrams, params['top_sumgram_count'], params=params )

    report['params'] = params
    report['created_at_utc'] = datetime.utcnow().isoformat().split('.')[0] + 'Z'
    report['top_sumgrams'] = top_ngrams[:params['top_sumgram_count']]

    '''
        After ranking documents, the doc_indx in 'ranked_sentences' and top_sumgrams.postings and top_sumgrams.parent_sentences
        is no longer valid because they reference the initial doc_lst permutation, so fix this with doc_id_new_doc_indx_map
    '''
    update_doc_indx(report, doc_id_new_doc_indx_map)
    fmt_report( report['top_sumgrams'], params ) #fmt_report() need to be called last since it potentially could modify merged_ngrams
    
    if( params['stanford_corenlp_server'] == False and params['sentence_tokenizer'] == 'ssplit' ):
        logger.info('\n\tStanford CoreNLP Server was OFF after an attempt to start it, so regex_get_sentences() was used to segment sentences.\n\tWe highly recommend you install and run it \n\t(see: https://ws-dl.blogspot.com/2018/03/2018-03-04-installing-stanford-corenlp.html)\n\tbecause Stanford CoreNLP does a better job segmenting sentences than regex.\n\tHowever, if you have no need to utilize sentence ranking, disregard this advise.\n')

    return report

def get_args():

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30))
    parser.add_argument('path', nargs='*', help='File(s) or path to file(s) or URL(s)')
    
    parser.add_argument('-d', '--print-details', help='Print detailed output', action='store_true')
    parser.add_argument('-m', '--max-ngram', help='The maximum length of sumgram generated', type=int, default=10)
    parser.add_argument('-n', '--base-ngram', help='The base n (integer) for generating top sumgrams, if n = 2, bigrams would be the base ngram', type=int, default=2)
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-s', '--sentences-rank-count', help='The count of top ranked sentences to generate', type=int, default=10)
    parser.add_argument('-t', '--top-sumgram-count', help='The count of top sumgrams to generate', type=int, default=10)
    
    parser.add_argument('--add-stopwords', nargs='+', help='Single or multiple additional stopwords or path to stopwords file (one stopword per line)', default=[])
    parser.add_argument('--boilerplate-rm-method', help='Method to apply for removing HTML boilerplate', choices=['boilerpy3.DefaultExtractor', 'boilerpy3.ArticleExtractor', 'boilerpy3.ArticleSentencesExtractor', 'boilerpy3.LargestContentExtractor', 'boilerpy3.CanolaExtractor', 'boilerpy3.KeepEverythingExtractor', 'boilerpy3.NumWordsRulesExtractor', 'nltk'], default='boilerpy3.ArticleExtractor')
    parser.add_argument('--collocations-pattern', help='User-defined regex rule to extract collocations for pos_glue_split_ngrams', default='')
    parser.add_argument('--corenlp-host', help='Stanford CoreNLP Server host (needed for decent sentence tokenizer)', default='localhost')
    parser.add_argument('--corenlp-port', help='Stanford CoreNLP Server port (needed for decent sentence tokenizer)', default='9000')
    parser.add_argument('--corenlp-max-sentence-words', help='Stanford CoreNLP maximum words per sentence', default=100)
    parser.add_argument('--include-postings', help='Include inverted index of term document mappings', action='store_true')#default is false except not included, in which case it's true
    parser.add_argument('--max-file-depth', help='When reading files recursively from directory stop at the specified path depth. 0 means no restriction', type=int, default=1)
    
    parser.add_argument('--log-file', help='Log output filename', default='')
    parser.add_argument('--log-format', help='Log print format, see: https://docs.python.org/3/howto/logging-cookbook.html', default='')
    parser.add_argument('--log-level', help='Log level', choices=['critical', 'error', 'warning', 'info', 'debug', 'notset'], default='info')
    
    parser.add_argument('--min-df', help='See min_df in https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html', default=0.01)
    parser.add_argument('--mvg-window-min-proper-noun-rate', help='Mininum rate threshold (larger, stricter) to consider a multi-word proper noun a candidate to replace an ngram', type=float, default=0.5)
    parser.add_argument('--ngram-printing-mw', help='Mininum width for printing ngrams', type=int, default=50)
    
    parser.add_argument('--base-ngram-ansi-color', help='Highlight (color code format - XXm, e.g., 91m) base ngram when printing top ngrams, set to empty string to switch off color', default='91m')
    parser.add_argument('--no-default-stopwords', help='Do not use default English stopwords list (default is False)', action='store_true')
    parser.add_argument('--no-mvg-window-glue-split-ngrams', help='Do not glue split top ngrams with Moving Window method (default is False)', action='store_true')
    parser.add_argument('--no-parent-sentences', help='Do not include sentences that mention top ngrams in top ngrams payload (default is False)', action='store_true')
    parser.add_argument('--no-pos-glue-split-ngrams', help='Do not glue split top ngrams with POS method (default is False)', action='store_true')
    parser.add_argument('--no-rank-sentences', help='Do not rank sentences flag (default is False)', action='store_true')
    parser.add_argument('--no-rank-docs', help='Do not rank documents flag (default is False)', action='store_true')

    parser.add_argument('--parallel-readtext', help='Read input files in parallel', action='store_true')
    parser.add_argument('--pos-glue-split-ngrams-coeff', help='Coeff. ([0, 1]) for permitting matched ngram replacement by pos_glue_split_ngrams(), bigger means stricter', type=float, default=0.5)
    parser.add_argument('--pretty-print', help='Pretty print JSON output', action='store_true')
    parser.add_argument('--rm-subset-top-ngrams-coeff', help='Coeff. ([0, 1]) for permitting matched ngram replacement by rm_subset_top_ngrams(), bigger means stricter', type=float, default=0.5)
    
    parser.add_argument('--sentence-pattern', help='For sentence ranking: Regex string that specifies tokens for sentence tokenization', default='[.?!][ \n]|\n+')
    parser.add_argument('--sentence-tokenizer', help='For sentence ranking: Method for segmenting sentences', choices=['ssplit', 'regex'], default='regex')
    parser.add_argument('--shift', help='Factor to shift top ngram calculation', type=int, default=0)
    parser.add_argument('--token-pattern', help='Regex string that specifies tokens for document tokenization', default=r'(?u)\b[a-zA-Z\'\’-]+[a-zA-Z]+\b|\d+[.,]?\d*')
    parser.add_argument('--title', help='Text label to be used as a heading when printing top sumgrams', default='')
    parser.add_argument('--thread-count', help='Maximum number of threads to use for parallel operations like segmenting sentences', type=int, default=5)
    parser.add_argument('--update-rate', help='Print 1 message per update-rate for long-running tasks', type=int, default=50)

    return parser

def get_default_args(user_params):
    #to be used by those who do not use this program from main, but call get_top_sumgrams directly
    parser = get_args()
    for key, val in parser._option_string_actions.items():
        
        if( val.default is None ):
            continue

        if( val.dest not in user_params ):
            user_params[val.dest] = val.default

    del user_params['help']
    user_params['add_stopwords_file'] = []
    return user_params


def proc_req(doc_lst, params):
    
    params.setdefault('print_details', False)
    report = get_top_sumgrams(doc_lst, params['base_ngram'], params)
    
    if( 'top_sumgrams' in report and params['print_details'] is False ):
        #since final top sumgrams not printed, print now
        print_top_ngrams( params['base_ngram'], report['top_sumgrams'], params['top_sumgram_count'], params=params )

    if( params['output'] is not None ):
        dumpJsonToFile( params['output'], report, indentFlag=params['pretty_print'], extraParams={'verbose': False} )
        print('wrote output:', params['output'])

def set_logger_dets(logger_dets):

    if( len(logger_dets) == 0 ):
        return

    console_handler = logging.StreamHandler()

    if( 'level' in logger_dets ):
        logger.setLevel( logger_dets['level'] )
    else:
        logger.setLevel( logging.INFO )

    if( 'file' in logger_dets ):
        logger_dets['file'] = logger_dets['file'].strip()
        
        if( logger_dets['file'] != '' ):
            file_handler = logging.FileHandler( logger_dets['file'] )
            proc_log_handler(file_handler, logger_dets)
    else:
        proc_log_handler(console_handler, logger_dets)
    
def proc_log_handler(handler, logger_dets):
    
    if( handler is None ):
        return
        
    if( 'level' in logger_dets ):
        handler.setLevel( logger_dets['level'] )    
        
        if( logger_dets['level'] == logging.ERROR ):
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s :\n%(message)s')
            handler.setFormatter(formatter)

    if( 'format' in logger_dets ):
        
        logger_dets['format'] = logger_dets['format'].strip()
        if( logger_dets['format'] != '' ):
            formatter = logging.Formatter( logger_dets['format'] )
            handler.setFormatter(formatter)

    logger.addHandler(handler)


def set_log_defaults(params):
    
    params['log_dets'] = {}

    if( params['log_level'] == '' ):
        params['log_dets']['level'] = logging.INFO
    else:
        
        log_levels = {
            'CRITICAL': 50,
            'ERROR': 40,
            'WARNING': 30,
            'INFO': 20,
            'DEBUG': 10,
            'NOTSET': 0
        }

        params['log_level'] = params['log_level'].strip().upper()

        if( params['log_level'] in log_levels ):
            params['log_dets']['level'] = log_levels[ params['log_level'] ]
        else:
            params['log_dets']['level'] = logging.INFO

        if( params['log_dets']['level'] == 10 ):#10: debug
            params['print_details'] = True
    
    params['log_format'] = params['log_format'].strip()
    params['log_file'] = params['log_file'].strip()

    if( params['log_format'] != '' ):
        params['log_dets']['format'] = params['log_format']

    if( params['log_file'] != '' ):
        params['log_dets']['file'] = params['log_file']


def main():

    if( len(sys.argv) > 1 and (sys.argv[1] == '-v' or sys.argv[1] == '--version') ):
        from sumgram import __version__
        print(__version__)
        return

    parser = get_args()
    args = parser.parse_args()
    params = vars(args)

    if( len(sys.argv) == 1 ):
        parser.print_help()
        return
    
    doc_lst = []
    set_log_defaults(params)
    set_logger_dets( params['log_dets'] )
    
    if( len(sys.argv) > 1 and (sys.argv[-1] == '-') ):
        try:
            doc_lst = [{'text': line} for line in sys.stdin]
        except:
            genericErrorInfo()

        params['add_stopwords'] = params['add_stopwords'][:-1] if (len(params['add_stopwords']) != 0 and params['add_stopwords'][-1].strip() == '-') else params['add_stopwords']
    else:
        doc_lst = generic_txt_extrator(args.path, max_file_depth=params['max_file_depth'], boilerplate_rm_method=params['boilerplate_rm_method'])
    
    '''
        add_stopwords:
        * unigrams in add_stopwords are used to complemented stopwords in getStopwordsSet() to build initial top n-ngram, see: CountVectorizer(stop_words=stopwords,...)
        * n-ngrams in add_stopwords are used in removing sumgrams that have n-ngrams, see: rm_empty_and_stopword_ngrams()
    '''
    add_stopwords = []
    params['add_stopwords_file'] = []
    for st in generic_txt_extrator(params['add_stopwords']):
        
        if( 'filename' in st ):
            params['add_stopwords_file'].append(st)
        else:
            add_stopwords.append(st)
    
    params['add_stopwords'] = get_user_stopwords(add_stopwords)
    params['add_stopwords_file'] = get_user_stopwords(params['add_stopwords_file'])
    params['referrer'] = 'main'
    proc_req(doc_lst, params)

if __name__ == 'sumgram.sumgram':
    from sumgram.util import dumpJsonToFile
    from sumgram.util import getColorTxt
    from sumgram.util import getStopwordsSet
    from sumgram.util import genericErrorInfo
    from sumgram.util import generic_txt_extrator
    from sumgram.util import isMatchInOrder
    from sumgram.util import nlpIsServerOn
    from sumgram.util import nlpSentenceAnnotate
    from sumgram.util import nlpServerStartStop
    from sumgram.util import overlapFor2Sets
    from sumgram.util import parallelTask
    from sumgram.util import phraseTokenizer
    from sumgram.util import rmStopwords
    from sumgram.util import sortDctByKey
else:
    from util import dumpJsonToFile
    from util import getColorTxt
    from util import getStopwordsSet
    from util import genericErrorInfo
    from util import generic_txt_extrator
    from util import isMatchInOrder
    from util import nlpIsServerOn
    from util import nlpSentenceAnnotate
    from util import nlpServerStartStop
    from util import overlapFor2Sets
    from util import parallelTask
    from util import phraseTokenizer
    from util import rmStopwords
    from util import sortDctByKey

    if __name__ == '__main__':    
        main()
