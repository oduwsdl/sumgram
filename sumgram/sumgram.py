import argparse
import copy
import numpy as np
import re
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer

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

	if( params['include_postings'] ):
		return

	for i in range(len(ngram_lst)):
		del ngram_lst[i]['postings']



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

		#ensure the splitting pattern corresponds to that used for ngrams
		sent_set = set( re.findall(params['token_pattern'], sentence.lower()) )
		ov = calc_avg_sentence_overlap( ngram_sentences, sent_set )
		
		sentences_lst.append({
			'avg_overlap': ov,
			'sentence': sentence,
			'doc_indx': doc_indx,
			'doc_id': doc_id,
			'sent_indx': i
		})
	
	return sentences_lst

def extract_doc_sentences(text, sentence_tokenizer, dedup_set, multi_word_proper_nouns, params=None):

	if( text == '' ):
		return []

	if( params is None ):
		params = {}

	params.setdefault('corenlp_host', 'localhost')
	params.setdefault('corenlp_port', '9000')

	filtered_sentences = []

	if( params['stanford_corenlp_server'] ):
		doc = nlpSentenceAnnotate(text.replace('\n', ' '), host=params['corenlp_host'], port=params['corenlp_port'])
	else:
		doc = {}
	
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
				
				if( tok_len > params['corenlp_max_sentence_words'] ):
					#this sentence is too long so force split 
					filtered_sentences += regex_get_sentences(sent['sentence'], sentence_tokenizer, dedup_set)
				else:
					filtered_sentences.append({ 'sentence': sentence, 'tok_len': tok_len })

	#if corenlp sentence segmentation is not used, use regex sentence segmentation
	if( len(filtered_sentences) == 0 ):
		filtered_sentences = regex_get_sentences(text, sentence_tokenizer, dedup_set)

	return filtered_sentences

def regex_get_sentences(text, sentence_tokenizer, dedup_set):

	if( text == '' ):
		return []

	sentences = re.split(sentence_tokenizer, text)
	filtered_sentences = []

	for sentence in sentences:
		
		sentence = sentence.replace('\n', ' ').strip()
		lowercase_sent = sentence.lower()
		if( sentence == '' ):
			continue
		
		if( lowercase_sent in dedup_set ):
			continue
	
		dedup_set.add(lowercase_sent)
		filtered_sentences.append({ 'sentence': sentence })

	return filtered_sentences


def get_ranked_docs(ngram_lst, doc_dct_lst):

	ranked_docs = {}
	N = len(ngram_lst)

	#print('\nget_ranked_docs(): N =', N)

	for i in range( len(ngram_lst) ):

		#print('\t\ti:', i)
		#print('\t\tt:', ngram_lst[i][0])

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
	return ranked_docs

def rank_sents_frm_top_ranked_docs(ngram_sentences, ranked_docs, all_doc_sentences, extra_params=None):

	if( extra_params is None ):
		extra_params = {}

	extra_params.setdefault('sentences_rank_count', 20)

	print('\nrank_sents_frm_top_ranked_docs():')

	all_top_ranked_docs_sentences = []
	for doc in ranked_docs:
		
		doc_indx = doc[0]
		doc_id = doc[1]['doc_id']
		all_top_ranked_docs_sentences += get_docs_sentence_score(ngram_sentences, all_doc_sentences[doc_indx], doc_indx, doc_id, extra_params)
	
	return sorted(all_top_ranked_docs_sentences, key=lambda x: x['avg_overlap'], reverse=True)[:extra_params['sentences_rank_count']]



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
			container[proper_noun_lower] = {'freq': 1, 'raw': proper_noun, 'nnp_rate': proper_noun_rate }

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

def get_sentence_match_ngram(ngram, ngram_toks, sentences, doc_indx):

	debug_verbose = False
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
				if( debug_verbose ):
					print('\nDID NOT FIND NGRAM POST SPLITTING' * 10)
					print('\tngram:', ngram)
					print('\tsentence:', sentence)
					print('\tsent_tok not printed')
					print()
				continue

			sentence_dets = {
				'ori_sent': ori_sent,
				'sent_indx': i,
				'doc_indx': doc_indx,
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
	sent_count = len(sentences)
	ngram = ngram.strip()

	if( sent_count == 1 or ngram == '' ):
		#it's possible for ngram = '', search for 'ngram_history'
		return ''

	window_size = 0
	max_sent_toks = 0
	final_multi_word_proper_noun = {}
	max_multiprpnoun_lrb = {}

	while True:

		window_size += 1
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

			if( params['debug_verbose'] ):
				print( '\n\twindow_size:', window_size )
				print( '\tngram:', ngram_toks )
				print( '\tngram in sent (start/length):', ngram_start, ngram_length )
				print( '\tsent keys:', sent.keys() )
				print( '\tori:', sent['ori_sent'] )
				print( '\tsent:', i, 'of', sent_count, ':', sent['toks'] )
				print( '\tsent_len:', sent_toks_count)

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
				
				if( params['debug_verbose'] ):
					print( '\t\t' + lrb + ':', multi_word_proper_noun_lrb )
			
			if( params['debug_verbose'] ):
				print( '\t\tsent_count:', sent_count )



		if( params['debug_verbose'] ):
			print('\n\twindow_size:', window_size, 'results:')


		#find multi-word proper noun with the highest frequency for left, right, and both sentence building policies
		for lrb, multiprpnoun_lrb in phrase_counts.items():
			
			multiprpnoun_lrb = sorted(multiprpnoun_lrb.items(), key=lambda x: x[1]['rate'], reverse=True)
			if( len(multiprpnoun_lrb) != 0 ):
				
				if( params['debug_verbose'] ):
					print('\t\tmax', lrb + ':', multiprpnoun_lrb[0])
				
				rate = multiprpnoun_lrb[0][1]['rate']
				if( rate > max_multiprpnoun_lrb[window_size]['rate'] ):
					max_multiprpnoun_lrb[window_size]['ngram'] = multiprpnoun_lrb[0][0]
					max_multiprpnoun_lrb[window_size]['rate'] = rate
					max_multiprpnoun_lrb[window_size]['lrb'] = lrb

		if( params['debug_verbose'] ):
			print('\tlast max for this window_size:', max_multiprpnoun_lrb[window_size])
			print('\tmax_sent_toks:', max_sent_toks)
			print()

		if( params['mvg_window_min_proper_noun_rate'] > max_multiprpnoun_lrb[window_size]['rate'] or window_size == max_sent_toks ):
			
			if( params['debug_verbose'] ):
				print("\tbreaking criteria reached: mvg_window_min_proper_noun_rate > max_multiprpnoun_lrb[window_size]['rate'] OR window_size (" + str(window_size) + ") == max_sent_toks (" + str(max_sent_toks) + ")")
				print('\tmvg_window_min_proper_noun_rate:', params['mvg_window_min_proper_noun_rate'])
				print("\tmax_multiprpnoun_lrb[window_size]['rate']:", max_multiprpnoun_lrb[window_size]['rate'])			
		
			break

	while( window_size != 0 ):
		#get best match longest multi-word ngram 
		if( max_multiprpnoun_lrb[window_size]['rate'] >= params['mvg_window_min_proper_noun_rate'] ):
			final_multi_word_proper_noun['proper_noun'] = max_multiprpnoun_lrb[window_size]['ngram']
			final_multi_word_proper_noun['rate'] = max_multiprpnoun_lrb[window_size]['rate']

			if( params['debug_verbose'] ):
				print('\tfinal winning max:', max_multiprpnoun_lrb[window_size])
				print('\twindow_size:', window_size)

			break

		window_size -= 1

	return final_multi_word_proper_noun

def pos_glue_split_ngrams(top_ngrams, k, pos_glue_split_ngrams_coeff, ranked_multi_word_proper_nouns, params):

	if( pos_glue_split_ngrams_coeff == 0 ):
		pos_glue_split_ngrams_coeff = 1

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

				if( ngram == multi_word_proper_noun or	mult_wd_prpnoun[1]['freq'] < top_ngrams[i]['term_freq']/pos_glue_split_ngrams_coeff ):
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
						'cur_freq': mult_wd_prpnoun[1]['freq']
					}

					if( multi_word_proper_noun in multi_word_proper_noun_dedup_set ):
						top_ngrams[i]['ngram'] = ''
						new_ngram_dct['cur_ngram'] = ''
					else:
						top_ngrams[i]['ngram'] = multi_word_proper_noun
						new_ngram_dct['cur_ngram'] = multi_word_proper_noun
						
						multi_word_proper_noun_dedup_set.add(multi_word_proper_noun)

					top_ngrams[i].setdefault('ngram_history', [])
					top_ngrams[i]['ngram_history'].append(new_ngram_dct)
				
				break

def mvg_window_glue_split_ngrams(top_ngrams, k, all_doc_sentences, params=None):

	print('\nmvg_window_glue_split_ngrams():')
	if( params is None ):
		params = {}

	#params['debug_verbose'] = True

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

		if( params['debug_verbose'] ):
			print('\t', i, 'ngram:', ngram)

		for doc_dct in top_ngrams[i]['postings']:
			
			doc_indx = doc_dct['doc_indx']
			phrase_cands += get_sentence_match_ngram( ngram, ngram_toks, all_doc_sentences[doc_indx], doc_indx )

		for phrase in phrase_cands:
			phrase_cands_minus_toks.append({
				'sentence': phrase['ori_sent'],
				'sent_indx': phrase['sent_indx'],
				'doc_indx': phrase['doc_indx']
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
				new_ngram_dct['proper_noun_rate'] = multi_word_proper_noun['rate']

				multi_word_proper_noun_dedup_set.add( multi_word_proper_noun['proper_noun'] )

			top_ngrams[i].setdefault('ngram_history', [])
			top_ngrams[i]['ngram_history'].append(new_ngram_dct)

		top_ngrams[i]['parent_sentences'] = phrase_cands_minus_toks
			
		if( params['debug_verbose'] ):
			print('*' * 200)
			print('*' * 200)
			print()



def rm_empty_ngrams(top_ngrams, k):

	final_top_ngrams = []

	for i in range( len(top_ngrams) ):
		
		if( i == k - 1 ):
			break 

		if( top_ngrams[i]['ngram'] == '' ):
			continue

		final_top_ngrams.append( top_ngrams[i] )

	return final_top_ngrams

def rm_subset_top_ngrams(top_ngrams, k, rm_subset_top_ngrams_coeff, params):

	if( rm_subset_top_ngrams_coeff == 0 ):
		rm_subset_top_ngrams_coeff = 1

	debug_verbose = False
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
					if( top_ngrams[parent_indx]['term_freq'] >= top_ngrams[child_indx]['term_freq']/rm_subset_top_ngrams_coeff ):

						if( top_ngrams[parent_indx]['adopted_child'] == False ):
							
							top_ngrams[parent_indx]['adopted_child'] = True
							top_ngrams[child_indx]['ngram'] = parent_ngram_cand

							new_ngram_dct = {
								'prev_ngram': child_ngram_cand,
								'cur_ngram': parent_ngram_cand,
								'annotator': 'subset'
							}
							top_ngrams[child_indx].setdefault('ngram_history', [])
							top_ngrams[child_indx]['ngram_history'].append(new_ngram_dct)

							if( debug_verbose ):
								print('\teven though parent (' + str(parent_indx) + ') is in lower index than child:', child_indx)
								print('\treplacing child_ngram_cand:', '"' + child_ngram_cand + '" with parent_ngram_cand: "' + parent_ngram_cand + '"')
								print('\treplacing parent/child tf:', top_ngrams[parent_indx]['term_freq'], top_ngrams[child_indx]['term_freq'])
								print()
						else:
							
							top_ngrams[child_indx]['ngram'] = ''

							if( debug_verbose ):
								print('\teven though parent (' + str(parent_indx) + ') is in lower index than child:', child_indx)
								print('\twould have replaced child_ngram_cand:', '"' + child_ngram_cand + '" with parent_ngram_cand: "' + parent_ngram_cand + '"')
								print('\twould have replaced parent/child tf:', top_ngrams[parent_indx]['term_freq'], top_ngrams[child_indx]['term_freq'])
								print('\tbut this parent has already adopted a child so delete this child.')
								print()
					

				

	return top_ngrams
	
def print_top_ngrams(n, top_ngrams, top_ngram_count, params=None):

	if( params is None ):
		params = {}

	params.setdefault('ngram_printing_mw', 40)
	params.setdefault('title', '')

	mw = params['ngram_printing_mw']
	ngram_count = len(top_ngrams)

	print('Summary for', ngram_count, 'top n-grams (base n: ' + str(n) + '):')
	print()

	if( params['title'] != '' ):
		print( params['title'] )

	print( '{:^6} {:<{mw}} {:^6} {:<6}'.format('rank', 'ngram', 'TF', 'TF-Rate', mw=mw) )
	for i in range(top_ngram_count):
		
		if( i == ngram_count ):
			break
		
		ngram = top_ngrams[i]
		ngram_txt = ngram['ngram']
		if( len(ngram_txt) > mw ) :
			ngram_txt = ngram_txt[:mw-3] + '...'

		print( "{:^6} {:<{mw}} {:^6} {:^6}".format(i+1, ngram_txt, ngram['term_freq'], "{:.2f}".format(ngram['term_rate']), mw=mw) )
	print()

def extract_top_ngrams(doc_lst, doc_dct_lst, n, params):

	print('\nextract_top_ngrams(): token_pattern:', params['token_pattern'])
	
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
	stopwords = getStopwordsSet() | bif_stopwords['unigrams']

	count_vectorizer = CountVectorizer(stop_words=stopwords, token_pattern=params['token_pattern'], ngram_range=(n, n), binary=binary_tf_flag)
	try:
		#tf_matrix is a binary TF matrix if doc_lst.len > 1, non-binary otherwise
		tf_matrix = count_vectorizer.fit_transform(doc_lst).toarray()
	except:
		genericErrorInfo()
		return []

	#every entry in list top_ngrams is of type: (a, b), a: term, b: term position in TF matrix
	
	top_ngrams = count_vectorizer.get_feature_names()
	filtered_top_ngrams = []
	total_freq = 0
	
	for i in range(tf_matrix.shape[1]):

		if( top_ngrams[i] in bif_stopwords['multigrams'] ):
			continue
		
		matrix_row = tf_matrix[:,i]
		if( binary_tf_flag ):
			row_sum_tf = np.count_nonzero(matrix_row)#row_sum_tf count (TF) of documents with non-zero entries
		else:
			row_sum_tf = int(matrix_row[0])
		
		#select documents with non-zero entries for term, doc index begins at 1
		non_zero_docs = np.flatnonzero(matrix_row)#non_zero_docs: list of index positions of documents with nonzero entry for vocabulary at i
		
		#find a simpler way to convert Int64 to native int
		postings = []
		for doc_indx in non_zero_docs:
			
			doc_indx = int(doc_indx)
			postings.append({
				'doc_indx': doc_indx, #Int64 to native int
				'doc_id': doc_dct_lst[doc_indx]['doc_id'],
				'doc_details': fmt_posting( doc_dct_lst[doc_indx] )
			})
		

		filtered_top_ngrams.append( get_ngram_dct(top_ngrams[i], row_sum_tf, postings) )
		total_freq += filtered_top_ngrams[-1]['term_freq']

	if( doc_count == 1 ):
		N = total_freq
	else:
		N = doc_count

	for i in range(len(filtered_top_ngrams)):
		filtered_top_ngrams[i]['term_rate'] = filtered_top_ngrams[i]['term_freq']/N

	
	return sorted(filtered_top_ngrams, key=lambda ngramEntry: ngramEntry['term_freq'], reverse=True)

def get_user_stopwords(comma_sep_stopwords):

	comma_sep_stopwords = comma_sep_stopwords.strip()
	if( comma_sep_stopwords == '' ):
		return set()

	add_stopwords = comma_sep_stopwords.split(',')
	return set( [s.strip().lower() for s in add_stopwords] )

def get_top_ngrams(n, doc_dct_lst, params=None):
	
	print('\nget_top_ngram():')
	np.set_printoptions(threshold=np.nan, linewidth=120)

	if( params is None or isinstance(params, dict) == False ):
		params = {}
	
	report = {}
	if( len(doc_dct_lst) == 0 ):
		return report

	if( n < 1 ):
		n = 1

	params = get_default_args(params)
	
	params['add_stopwords'] = get_user_stopwords( params['add_stopwords'] ) 
	params.setdefault('binary_tf_flag', True)#Multiple occurrence of term T in a document counts as 1, TF = total number of times term appears in collection
	params['stanford_corenlp_server'] = nlpIsServerOn()

	if( params['stanford_corenlp_server'] == False ):
		print('\n\tAttempting to start Stanford CoreNLP Server (we need it to segment sentences)\n')
		nlpServerStartStop('start')
		params['stanford_corenlp_server'] = nlpIsServerOn()
	

	doc_lst = []
	#doc_dct_lst: {doc_id: , text: }
	all_doc_sentences = {}
	multi_word_proper_nouns = {}
	dedup_set = set()
	for i in range(len(doc_dct_lst)):
		
		doc_dct_lst[i].setdefault('doc_id', i)
		doc_lst.append( doc_dct_lst[i]['text'] )

		#placing sentences inside doc_dct_lst[i] accounted for more runtime overhead
		all_doc_sentences[i] = extract_doc_sentences( 
			doc_dct_lst[i]['text'], 
			params['sentence_tokenizer'], 
			dedup_set, 
			multi_word_proper_nouns, 
			params=params
		)

		del doc_dct_lst[i]['text'] 

	multi_word_proper_nouns = rank_proper_nouns(multi_word_proper_nouns)
	print('\tdone adding sentences')
	print('\tshift:', params['shift'])
		
	top_ngrams = extract_top_ngrams(doc_lst, doc_dct_lst, n, params)

	if( len(top_ngrams) == 0 ):
		return report
	
	if( params['top_ngram_count'] < 1 or params['top_ngram_count'] > len(top_ngrams) ):
		params['top_ngram_count'] = len(top_ngrams)
	
	'''
		shifting is off by default
		shifting is done in an attempt to perform ngram summary on non-top terms 
		this may be required for comparing two different collections that have similar top terms
		so shifting is an attempt to perform process non-top terms in order to find distinguishing ngrams below the top ngrams
	'''
	shift_factor = params['shift'] * params['top_ngram_count']	
	if( shift_factor >= len(top_ngrams) ):
		shift_factor = 0

	print('\ttop_ngrams.len:', len(top_ngrams))
	if( shift_factor > 0 ):
		params['top_ngram_shift_factor'] = shift_factor
		top_ngrams = top_ngrams[shift_factor:]
		print('\ttop_ngrams.post shift len:', len(top_ngrams))


	doc_count = len(doc_dct_lst)
	if( doc_count == 1 ):

		N = len(top_ngrams)
		params['tf_label'] = 'Single Document Term Frequency'
		params['binary_tf_flag'] = False

	else:

		N = doc_count
		params['tf_label'] = 'Collection Term Frequency (1 term count per document)'

	params['tf_normalizing_divisor'] = N
	report = { 'n': n, 'top_ngram_count': params['top_ngram_count']}

	print('\tdoc_lst.len:', doc_count)
	print('\ntop ngrams before finding multi-word proper nouns:')
	print_top_ngrams( n, top_ngrams, params['top_ngram_count'], params=params )
	
	if( params['no_pos_glue_split_ngrams'] == False ):
		pos_glue_split_ngrams( top_ngrams, params['top_ngram_count'] * 2, params['pos_glue_split_ngrams_coeff'], multi_word_proper_nouns, params )

	#subset top_ngrams will be replace with their supersets, thus shrinking top_ngram_count counts after this operation, so maximize the chances of reporting user-supplied c, begin by processing: top_ngram_count * 2
	if( params['no_mvg_window_glue_split_ngrams'] == False ):
		mvg_window_glue_split_ngrams( top_ngrams, params['top_ngram_count'] * 2, all_doc_sentences, params=params )
	
	print('\ntop ngrams after finding multi-word proper nouns:')
	print_top_ngrams( n, top_ngrams, params['top_ngram_count'], params=params )
	
	top_ngrams = rm_subset_top_ngrams( top_ngrams, params['top_ngram_count'] * 2, params['rm_subset_top_ngrams_coeff'], params )
	print('\ntop ngrams after removing subset phrases:')
	print_top_ngrams( n, top_ngrams, params['top_ngram_count'], params=params )

	top_ngrams = rm_empty_ngrams( top_ngrams, params['top_ngram_count'] * 2 )

	if( params['no_rank_docs'] == False ):
		report['ranked_docs'] = get_ranked_docs( top_ngrams, doc_dct_lst )

		if( params['sentences_rank_count'] > 0 and params['no_rank_sentences'] == False ):
			ngram_sentences = combine_ngrams( top_ngrams[:params['top_ngram_count']] )
			report['ranked_sentences'] = rank_sents_frm_top_ranked_docs( ngram_sentences, report['ranked_docs'], all_doc_sentences, params )

	
	report['top_ngrams'] = top_ngrams[:params['top_ngram_count']]
	print('\ntop ngrams after shifting empty slots:')
	print_top_ngrams( n, top_ngrams, params['top_ngram_count'], params=params )

	#fmt_report() need to be called last since it potentially could modify merged_ngrams
	fmt_report( report['top_ngrams'], params )
	report['params'] = params
	report['params']['collection_doc_count'] = doc_count

	if( params['stanford_corenlp_server'] == False ):
		print('\n\tStanford CoreNLP Server was OFF after an attempt to start it, so regex_get_sentences() was used to segment sentences.\n\tWe highly recommend you install and run it \n\t(see: https://ws-dl.blogspot.com/2018/03/2018-03-04-installing-stanford-corenlp.html)\n\tbecause Stanford CoreNLP does a better job segmenting sentences than regex.\n')

	return report

def get_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='Folder path containing input documents or path to single file')
	
	parser.add_argument('-n', help='The base n (integer) for generating top ngrams, if n = 2, bigrams would be the base ngram', type=int, default=2)
	parser.add_argument('-o', '--output', help='Output file')
	parser.add_argument('-s', '--sentences-rank-count', help='The count of top ranked sentences to generate', type=int, default=10)	
	parser.add_argument('-t', '--top-ngram-count', help='The count of top ngrams to generate', type=int, default=10)
	
	parser.add_argument('--add-stopwords', help='Comma-separated list of addition stopwords', default='')
	parser.add_argument('--corenlp-host', help='Stanford CoreNLP Server host (needed for decent sentence tokenizer)', default='localhost')
	parser.add_argument('--corenlp-port', help='Stanford CoreNLP Server port (needed for decent sentence tokenizer)', default='9000')
	parser.add_argument('--corenlp-max-sentence-words', help='Stanford CoreNLP maximum words per sentence', default=100)
	parser.add_argument('--debug-verbose', help='Print statements needed for debugging purpose', action='store_true')
	parser.add_argument('--include-postings', help='Include inverted index of term document mappings', action='store_true')#default is false except not included, in which case it's true
	parser.add_argument('--mvg-window-min-proper-noun-rate', help='Mininum rate threshold (larger, stricter) to consider a multi-word proper noun a candidate to replace an ngram', type=float, default=0.5)
	parser.add_argument('--ngram-printing-mw', help='Mininum width for printing ngrams', type=int, default=50)
	parser.add_argument('--no-rank-docs', help='Do not rank documents flag (default is True)', action='store_true')
	parser.add_argument('--no-rank-sentences', help='Do not rank sentences flag (default is True)', action='store_true')
	
	parser.add_argument('--no-pos-glue-split-ngrams', help='Do not glue split top ngrams with POS method (default is True)', action='store_true')
	parser.add_argument('--no-mvg-window-glue-split-ngrams', help='Do not glue split top ngrams with MOVING WINDOW method (default is True)', action='store_true')

	parser.add_argument('--pos-glue-split-ngrams-coeff', help='Coeff for permitting matched ngram replacement. Interpreted as 1/coeff', type=int, default=2)
	parser.add_argument('--pretty-print', help='Pretty print JSON output', action='store_true')
	parser.add_argument('--rm-subset-top-ngrams-coeff', help='Coeff. for permitting matched ngram replacement. Interpreted as 1/coeff', type=int, default=2)
	
	parser.add_argument('--sentence-tokenizer', help='For sentence ranking: Regex string that specifies tokens for sentence tokenization', default='[.?!][ \n]|\n+')
	parser.add_argument('--shift', help='Factor to shift top ngram calculation', type=int, default=0)
	parser.add_argument('--token-pattern', help='Regex string that specifies tokens for document tokenization', default=r'(?u)\b[a-zA-Z\'\’-]+[a-zA-Z]+\b|\d+[.,]?\d*')
	parser.add_argument('--title', help='Text label to be used as a heading when printing top ngrams', default='')

	return parser

def get_default_args(user_params):
	#to be used by those who do not use this program from main, but call get_top_ngrams directly
	parser = get_args()
	for key, val in parser._option_string_actions.items():
		
		if( val.default is None ):
			continue

		if( val.dest not in user_params ):
			user_params[val.dest] = val.default

	del user_params['help']
	return user_params


def proc_req(doc_lst, params):
	report = get_top_ngrams(params['n'], doc_lst, params)
	if( params['output'] is not None ):
		dumpJsonToFile( params['output'], report, indentFlag=params['pretty_print'] )
	
def main():
	parser = get_args()

	args = parser.parse_args()
	params = vars(args)
	
	doc_lst = getText(args.path)
	proc_req(doc_lst, params)


if __name__ == 'sumgram.sumgram':
	from sumgram.util import dumpJsonToFile
	from sumgram.util import getStopwordsSet
	from sumgram.util import genericErrorInfo
	from sumgram.util import getText
	from sumgram.util import isMatchInOrder
	from sumgram.util import nlpIsServerOn
	from sumgram.util import nlpSentenceAnnotate
	from sumgram.util import nlpServerStartStop
	from sumgram.util import overlapFor2Sets
	from sumgram.util import parallelTask
	from sumgram.util import phraseTokenizer
	from sumgram.util import readTextFromFile
	from sumgram.util import rmStopwords
	from sumgram.util import sortDctByKey
else:
	from util import dumpJsonToFile
	from util import getStopwordsSet
	from util import genericErrorInfo
	from util import getText
	from util import isMatchInOrder
	from util import nlpIsServerOn
	from util import nlpSentenceAnnotate
	from util import nlpServerStartStop
	from util import overlapFor2Sets
	from util import parallelTask
	from util import phraseTokenizer
	from util import readTextFromFile
	from util import rmStopwords
	from util import sortDctByKey

	if __name__ == '__main__':	
		main()
