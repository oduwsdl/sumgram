# sumgram

sumgram is a tool that
	Backbone function call stack
		get_top_ngrams()
			extract_doc_sentences()
			rank_proper_nouns()
			extract_top_ngrams()
			pos_glue_split_ngrams()
			mvg_window_glue_split_phrases()
			rm_subset_top_ngrams()
			optional: get_ranked_docs() ("--no-rank-docs" from command line or "no_rank_docs" from code)
			optional: rank_sents_frm_top_ranked_docs() ("--no-rank-sentences" from command line or "no_rank_sentences")
	
	extract_doc_sentences()
		- Segments sentences with stanford corenlp sentence segmenter (if active) else regex sentence segmenter
		- Populates multi_word_proper_nouns with multi-word proper nouns (only if stanford corenlp sentence segmenter is used). This is done by extract_proper_nouns()

	rank_proper_nouns()
		- Ranks (higher score, better) multi-word proper nouns score calculated by: freq * nnp_rate
		For example here are two proper nouns and their respective scores
			* "Centers for Disease Control" (NNP IN NNP NNP), freq: 50, nnp_rate: 3/4, score: 37.5
			* "Federal Emergency Management Agency" (NNP NNP NNP NNP), freq: 50, nnp_rate: 4/4, score: 50
		The rationale is to favor True Positive multi-word proper nouns. A multi-word proper noun with exclusively NNP types has a high probability of actually being a multi-word proper noun

	extract_top_ngrams()
		- Responsible for generating raw top ngrams of format
		    [
			    {
			        "ngram": "hurricane harvey",
			        "term_freq": 18,
			        "postings": [
			            {
			                "doc_indx": 0,
			                "doc_id": 0,
			                "doc_details": {
			                    "f": "./GetTopNgrams_testdocs/plaintext/small/cce48972c398e326a65f9fccb26ab3c7.txt"
			                }
			            },
			            {
			                "doc_indx": 1,
			                "doc_id": 1,
			                "doc_details": {
			                    "f": "./GetTopNgrams_testdocs/plaintext/small/9b89dde57a4ed2556c016f9743061cc1.txt"
			                }
			            },...
			        ],
			        "term_rate": 0.9
			    },...
		    ]

	pos_glue_split_ngrams()
		- First measure to merge split multi-word ngrams:
		For example this top ngram child "emergency management" was extracted (base ngram = 2) from its parent multi-word proper noun (mwpn):
		"federal emergency management agency". This function attempts to replace the child with the parent multi-word proper noun
		- Sensitivity (smaller, stricter) controlled by pos_glue_split_ngrams_coeff

	mvg_window_glue_split_phrases()
		- Second measure to merge split multi-word ngrams, main logic captured by rank_mltwd_proper_nouns()
		- Process summary: 
			For all sentences (ori in Process example) that encompass the split ngram ("emergency management") 
			
				- extract window_size term(s) from left of the split ngram and add to the left of the split ngram
				- extract window_size term(s) from right of the split ngram and add to the right of the split ngram
				- extract window_size term(s) from both left and right of the split ngram and add left to the left of the split ngram, and right to the right
			
				For example, given split ngram "emergency management", given window_size = 1, given original sentence tokens: ['more', 'than', '32', '000', 'people', 'have', 'been', 'housed', 'in', 'shelters', '', 'and', 'the', 'federal', 'emergency', 'management', 'agency', 'is', 'expecting', 'nearly', 'a', 'half', 'million', 'people', 'to', 'seek', 'some', 'sort', 'of', 'disaster', 'aid.']
				- left + split ngram: "federal" + "emergency management" = "federal emergency management"
				- split ngram + right: "emergency management" + "agency" = "emergency management agency"
				- left + split ngram + right: "federal" + "emergency management" + "agency" = "federal emergency management agency"

			For a given window_size, the winning mwpn is the one with the highest rate of occurrence
			If for a given window_size, the occurrence rate of the winning mwpn >= mvg_window_min_proper_noun_rate increment window_size and continue 
			until the winning mwpn freq. < mvg_window_min_proper_noun_rate.

			There could be multiple mwpn candidates that could potentially replace the split multi-word ngram.
			For example (Process example) for window_size 1, mwpn: "federal emergency management" with occurrence rate: 0.875
										  for window_size 2, mwpn: "the federal emergency management" with occurrence rate: 0.875
										  for window_size 3, mwpn: "the federal emergency management" with occurrence rate: 0.875

			Therefore, select the mwpn with largest window_size and with rate >= mvg_window_min_proper_noun_rate. Traverse candidate list in reverse order

		- Process example (trying to go from split top ngram "emergency management" to multi-word proper noun ngram "the federal emergency management"):

			1 ngram: emergency management

			WINDOW_SIZE 1
				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 14 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: More than 32,000 people have been housed in shelters, and the Federal Emergency Management Agency is expecting nearly a half million people to seek some sort of disaster aid.
				sent: 0 of 8 : ['more', 'than', '32', '000', 'people', 'have', 'been', 'housed', 'in', 'shelters', '', 'and', 'the', 'federal', 'emergency', 'management', 'agency', 'is', 'expecting', 'nearly', 'a', 'half', 'million', 'people', 'to', 'seek', 'some', 'sort', 'of', 'disaster', 'aid.']
				sent_len: 31
					left: federal emergency management
					right: emergency management agency
					both: federal emergency management agency
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 25 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: He said the bill contains the amount of Hurricane Harvey funding requested by the White House, which includes $7.4 billion for the Federal Emergency Management Agency disaster relief fund.
				sent: 1 of 8 : ['he', 'said', 'the', 'bill', 'contains', 'the', 'amount', 'of', 'hurricane', 'harvey', 'funding', 'requested', 'by', 'the', 'white', 'house', '', 'which', 'includes', '', '7.4', 'billion', 'for', 'the', 'federal', 'emergency', 'management', 'agency', 'disaster', 'relief', 'fund.']
				sent_len: 31
					left: federal emergency management
					right: emergency management agency
					both: federal emergency management agency
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 24 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: Here’s the latest: • With record floodwaters, more than 450,000 people are likely to seek federal aid, the Federal Emergency Management Agency said on Monday.
				sent: 2 of 8 : ['here’s', 'the', 'latest', '', '', '', 'with', 'record', 'floodwaters', '', 'more', 'than', '450', '000', 'people', 'are', 'likely', 'to', 'seek', 'federal', 'aid', '', 'the', 'federal', 'emergency', 'management', 'agency', 'said', 'on', 'monday.']
				sent_len: 30
					left: federal emergency management
					right: emergency management agency
					both: federal emergency management agency
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 31 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: Managing the shelters that are housing tens of thousands of them remains “the biggest battle that we have right now,” Brock Long, the administrator of the Federal Emergency Management Agency, said on Thursday.
				sent: 3 of 8 : ['managing', 'the', 'shelters', 'that', 'are', 'housing', 'tens', 'of', 'thousands', 'of', 'them', 'remains', '', 'the', 'biggest', 'battle', 'that', 'we', 'have', 'right', 'now', '', '', 'brock', 'long', '', 'the', 'administrator', 'of', 'the', 'federal', 'emergency', 'management', 'agency', '', 'said', 'on', 'thursday.']
				sent_len: 38
					left: federal emergency management
					right: emergency management agency
					both: federal emergency management agency
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 36 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: More than 50 people were evacuated from a Nashville neighborhood due to flooding from Harvey, but no deaths or injuries have been reported, according to a statement Friday morning from the Nashville Office of Emergency Management.
				sent: 4 of 8 : ['more', 'than', '50', 'people', 'were', 'evacuated', 'from', 'a', 'nashville', 'neighborhood', 'due', 'to', 'flooding', 'from', 'harvey', '', 'but', 'no', 'deaths', 'or', 'injuries', 'have', 'been', 'reported', '', 'according', 'to', 'a', 'statement', 'friday', 'morning', 'from', 'the', 'nashville', 'office', 'of', 'emergency', 'management.']
				sent_len: 38
					left: of emergency management
					right: emergency management
					both: of emergency management
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 10 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: The state also is requesting further assistance from the Federal Emergency Management Agency.
				sent: 5 of 8 : ['the', 'state', 'also', 'is', 'requesting', 'further', 'assistance', 'from', 'the', 'federal', 'emergency', 'management', 'agency.']
				sent_len: 13
					left: federal emergency management
					right: emergency management agency.
					both: federal emergency management agency.
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 8 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: City officials have been working with the Federal Emergency Management Agency, Texas leaders and employees from other cities and states to assess the damage caused by the storm, clear debris and look to start the rebuilding process.
				sent: 6 of 8 : ['city', 'officials', 'have', 'been', 'working', 'with', 'the', 'federal', 'emergency', 'management', 'agency', '', 'texas', 'leaders', 'and', 'employees', 'from', 'other', 'cities', 'and', 'states', 'to', 'assess', 'the', 'damage', 'caused', 'by', 'the', 'storm', '', 'clear', 'debris', 'and', 'look', 'to', 'start', 'the', 'rebuilding', 'process.']
				sent_len: 39
					left: federal emergency management
					right: emergency management agency
					both: federal emergency management agency
					sent_count: 8

				window_size: 1
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 21 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: About 21,000 households are living in 2,000 hotels in 33 states, according to Tony Robinson, the Federal Emergency Management Agency’s regional director.
				sent: 7 of 8 : ['about', '21', '000', 'households', 'are', 'living', 'in', '2', '000', 'hotels', 'in', '33', 'states', '', 'according', 'to', 'tony', 'robinson', '', 'the', 'federal', 'emergency', 'management', 'agency’s', 'regional', 'director.']
				sent_len: 26
					left: federal emergency management
					right: emergency management agency’s
					both: federal emergency management agency’s
					sent_count: 8

				window_size: 1 results:
					max left: ('federal emergency management', {'freq': 7, 'rate': 0.875})
					max right: ('emergency management agency', {'freq': 5, 'rate': 0.625})
					max both: ('federal emergency management agency', {'freq': 5, 'rate': 0.625})
				last max for this window_size: {'lrb': 'left', 'ngram': 'federal emergency management', 'rate': 0.875}
				max_sent_toks: 39

			WINDOW_SIZE 2
				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 14 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: More than 32,000 people have been housed in shelters, and the Federal Emergency Management Agency is expecting nearly a half million people to seek some sort of disaster aid.
				sent: 0 of 8 : ['more', 'than', '32', '000', 'people', 'have', 'been', 'housed', 'in', 'shelters', '', 'and', 'the', 'federal', 'emergency', 'management', 'agency', 'is', 'expecting', 'nearly', 'a', 'half', 'million', 'people', 'to', 'seek', 'some', 'sort', 'of', 'disaster', 'aid.']
				sent_len: 31
					left: the federal emergency management
					right: emergency management agency is
					both: the federal emergency management agency is
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 25 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: He said the bill contains the amount of Hurricane Harvey funding requested by the White House, which includes $7.4 billion for the Federal Emergency Management Agency disaster relief fund.
				sent: 1 of 8 : ['he', 'said', 'the', 'bill', 'contains', 'the', 'amount', 'of', 'hurricane', 'harvey', 'funding', 'requested', 'by', 'the', 'white', 'house', '', 'which', 'includes', '', '7.4', 'billion', 'for', 'the', 'federal', 'emergency', 'management', 'agency', 'disaster', 'relief', 'fund.']
				sent_len: 31
					left: the federal emergency management
					right: emergency management agency disaster
					both: the federal emergency management agency disaster
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 24 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: Here’s the latest: • With record floodwaters, more than 450,000 people are likely to seek federal aid, the Federal Emergency Management Agency said on Monday.
				sent: 2 of 8 : ['here’s', 'the', 'latest', '', '', '', 'with', 'record', 'floodwaters', '', 'more', 'than', '450', '000', 'people', 'are', 'likely', 'to', 'seek', 'federal', 'aid', '', 'the', 'federal', 'emergency', 'management', 'agency', 'said', 'on', 'monday.']
				sent_len: 30
					left: the federal emergency management
					right: emergency management agency said
					both: the federal emergency management agency said
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 31 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: Managing the shelters that are housing tens of thousands of them remains “the biggest battle that we have right now,” Brock Long, the administrator of the Federal Emergency Management Agency, said on Thursday.
				sent: 3 of 8 : ['managing', 'the', 'shelters', 'that', 'are', 'housing', 'tens', 'of', 'thousands', 'of', 'them', 'remains', '', 'the', 'biggest', 'battle', 'that', 'we', 'have', 'right', 'now', '', '', 'brock', 'long', '', 'the', 'administrator', 'of', 'the', 'federal', 'emergency', 'management', 'agency', '', 'said', 'on', 'thursday.']
				sent_len: 38
					left: the federal emergency management
					right: emergency management agency
					both: the federal emergency management agency
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 36 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: More than 50 people were evacuated from a Nashville neighborhood due to flooding from Harvey, but no deaths or injuries have been reported, according to a statement Friday morning from the Nashville Office of Emergency Management.
				sent: 4 of 8 : ['more', 'than', '50', 'people', 'were', 'evacuated', 'from', 'a', 'nashville', 'neighborhood', 'due', 'to', 'flooding', 'from', 'harvey', '', 'but', 'no', 'deaths', 'or', 'injuries', 'have', 'been', 'reported', '', 'according', 'to', 'a', 'statement', 'friday', 'morning', 'from', 'the', 'nashville', 'office', 'of', 'emergency', 'management.']
				sent_len: 38
					left: office of emergency management
					right: emergency management
					both: office of emergency management
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 10 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: The state also is requesting further assistance from the Federal Emergency Management Agency.
				sent: 5 of 8 : ['the', 'state', 'also', 'is', 'requesting', 'further', 'assistance', 'from', 'the', 'federal', 'emergency', 'management', 'agency.']
				sent_len: 13
					left: the federal emergency management
					right: emergency management agency.
					both: the federal emergency management agency.
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 8 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: City officials have been working with the Federal Emergency Management Agency, Texas leaders and employees from other cities and states to assess the damage caused by the storm, clear debris and look to start the rebuilding process.
				sent: 6 of 8 : ['city', 'officials', 'have', 'been', 'working', 'with', 'the', 'federal', 'emergency', 'management', 'agency', '', 'texas', 'leaders', 'and', 'employees', 'from', 'other', 'cities', 'and', 'states', 'to', 'assess', 'the', 'damage', 'caused', 'by', 'the', 'storm', '', 'clear', 'debris', 'and', 'look', 'to', 'start', 'the', 'rebuilding', 'process.']
				sent_len: 39
					left: the federal emergency management
					right: emergency management agency
					both: the federal emergency management agency
					sent_count: 8

				window_size: 2
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 21 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: About 21,000 households are living in 2,000 hotels in 33 states, according to Tony Robinson, the Federal Emergency Management Agency’s regional director.
				sent: 7 of 8 : ['about', '21', '000', 'households', 'are', 'living', 'in', '2', '000', 'hotels', 'in', '33', 'states', '', 'according', 'to', 'tony', 'robinson', '', 'the', 'federal', 'emergency', 'management', 'agency’s', 'regional', 'director.']
				sent_len: 26
					left: the federal emergency management
					right: emergency management agency’s regional
					both: the federal emergency management agency’s regional
					sent_count: 8

				window_size: 2 results:
					max left: ('the federal emergency management', {'freq': 7, 'rate': 0.875})
					max right: ('emergency management agency', {'freq': 2, 'rate': 0.25})
					max both: ('the federal emergency management agency', {'freq': 2, 'rate': 0.25})
				last max for this window_size: {'lrb': 'left', 'ngram': 'the federal emergency management', 'rate': 0.875}
				max_sent_toks: 39

			WINDOW_SIZE 3
				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 14 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: More than 32,000 people have been housed in shelters, and the Federal Emergency Management Agency is expecting nearly a half million people to seek some sort of disaster aid.
				sent: 0 of 8 : ['more', 'than', '32', '000', 'people', 'have', 'been', 'housed', 'in', 'shelters', '', 'and', 'the', 'federal', 'emergency', 'management', 'agency', 'is', 'expecting', 'nearly', 'a', 'half', 'million', 'people', 'to', 'seek', 'some', 'sort', 'of', 'disaster', 'aid.']
				sent_len: 31
					left: and the federal emergency management
					right: emergency management agency is expecting
					both: and the federal emergency management agency is expecting
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 25 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: He said the bill contains the amount of Hurricane Harvey funding requested by the White House, which includes $7.4 billion for the Federal Emergency Management Agency disaster relief fund.
				sent: 1 of 8 : ['he', 'said', 'the', 'bill', 'contains', 'the', 'amount', 'of', 'hurricane', 'harvey', 'funding', 'requested', 'by', 'the', 'white', 'house', '', 'which', 'includes', '', '7.4', 'billion', 'for', 'the', 'federal', 'emergency', 'management', 'agency', 'disaster', 'relief', 'fund.']
				sent_len: 31
					left: for the federal emergency management
					right: emergency management agency disaster relief
					both: for the federal emergency management agency disaster relief
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 24 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: Here’s the latest: • With record floodwaters, more than 450,000 people are likely to seek federal aid, the Federal Emergency Management Agency said on Monday.
				sent: 2 of 8 : ['here’s', 'the', 'latest', '', '', '', 'with', 'record', 'floodwaters', '', 'more', 'than', '450', '000', 'people', 'are', 'likely', 'to', 'seek', 'federal', 'aid', '', 'the', 'federal', 'emergency', 'management', 'agency', 'said', 'on', 'monday.']
				sent_len: 30
					left: the federal emergency management
					right: emergency management agency said on
					both: the federal emergency management agency said on
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 31 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: Managing the shelters that are housing tens of thousands of them remains “the biggest battle that we have right now,” Brock Long, the administrator of the Federal Emergency Management Agency, said on Thursday.
				sent: 3 of 8 : ['managing', 'the', 'shelters', 'that', 'are', 'housing', 'tens', 'of', 'thousands', 'of', 'them', 'remains', '', 'the', 'biggest', 'battle', 'that', 'we', 'have', 'right', 'now', '', '', 'brock', 'long', '', 'the', 'administrator', 'of', 'the', 'federal', 'emergency', 'management', 'agency', '', 'said', 'on', 'thursday.']
				sent_len: 38
					left: of the federal emergency management
					right: emergency management agency said
					both: of the federal emergency management agency said
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 36 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: More than 50 people were evacuated from a Nashville neighborhood due to flooding from Harvey, but no deaths or injuries have been reported, according to a statement Friday morning from the Nashville Office of Emergency Management.
				sent: 4 of 8 : ['more', 'than', '50', 'people', 'were', 'evacuated', 'from', 'a', 'nashville', 'neighborhood', 'due', 'to', 'flooding', 'from', 'harvey', '', 'but', 'no', 'deaths', 'or', 'injuries', 'have', 'been', 'reported', '', 'according', 'to', 'a', 'statement', 'friday', 'morning', 'from', 'the', 'nashville', 'office', 'of', 'emergency', 'management.']
				sent_len: 38
					left: nashville office of emergency management
					right: emergency management
					both: nashville office of emergency management
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 10 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: The state also is requesting further assistance from the Federal Emergency Management Agency.
				sent: 5 of 8 : ['the', 'state', 'also', 'is', 'requesting', 'further', 'assistance', 'from', 'the', 'federal', 'emergency', 'management', 'agency.']
				sent_len: 13
					left: from the federal emergency management
					right: emergency management agency.
					both: from the federal emergency management agency.
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 8 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: City officials have been working with the Federal Emergency Management Agency, Texas leaders and employees from other cities and states to assess the damage caused by the storm, clear debris and look to start the rebuilding process.
				sent: 6 of 8 : ['city', 'officials', 'have', 'been', 'working', 'with', 'the', 'federal', 'emergency', 'management', 'agency', '', 'texas', 'leaders', 'and', 'employees', 'from', 'other', 'cities', 'and', 'states', 'to', 'assess', 'the', 'damage', 'caused', 'by', 'the', 'storm', '', 'clear', 'debris', 'and', 'look', 'to', 'start', 'the', 'rebuilding', 'process.']
				sent_len: 39
					left: with the federal emergency management
					right: emergency management agency texas
					both: with the federal emergency management agency texas
					sent_count: 8

				window_size: 3
				ngram: ['emergency', 'management']
				ngram in sent (start/length): 21 2
				sent keys: dict_keys(['ori_sent', 'sent_indx', 'doc_indx', 'toks', 'ngram_start_indx', 'ngram_length'])
				ori: About 21,000 households are living in 2,000 hotels in 33 states, according to Tony Robinson, the Federal Emergency Management Agency’s regional director.
				sent: 7 of 8 : ['about', '21', '000', 'households', 'are', 'living', 'in', '2', '000', 'hotels', 'in', '33', 'states', '', 'according', 'to', 'tony', 'robinson', '', 'the', 'federal', 'emergency', 'management', 'agency’s', 'regional', 'director.']
				sent_len: 26
					left: the federal emergency management
					right: emergency management agency’s regional director.
					both: the federal emergency management agency’s regional director.
					sent_count: 8

				window_size: 3 results:
					max left: ('the federal emergency management', {'freq': 2, 'rate': 0.25})
					max right: ('emergency management agency is expecting', {'freq': 1, 'rate': 0.125})
					max both: ('and the federal emergency management agency is expecting', {'freq': 1, 'rate': 0.125})
				last max for this window_size: {'lrb': 'left', 'ngram': 'the federal emergency management', 'rate': 0.25}
				max_sent_toks: 39

				breaking criteria reached: mvg_window_min_proper_noun_rate > max_multiprpnoun_lrb[window_size]['rate'] OR window_size (3) == max_sent_toks (39)
				mvg_window_min_proper_noun_rate: 0.5
				max_multiprpnoun_lrb[window_size]['rate']: 0.25
				final winning max: {'lrb': 'left', 'ngram': 'the federal emergency management', 'rate': 0.875}
				window_size: 2


	rm_subset_top_ngrams()
		- Within the list of top ngrams sometimes some ngram might be a subset another, deconflict and keep one.
		For example, "category 4 hurricane" would be replaced by "a category 4 hurricane"
		- Sensitivity (smaller, stricter) controlled by rm_subset_top_ngrams_coeff

	get_ranked_docs()
		
		Given i ∈ N = |list of top ngrams|

		Give credit to documents that have highly ranked (bigger diff: N - i) terms in the ngram_lst 
		a document's score is awarded by accumulating the points awarded by the position of terms in the ngram_lst.
		Documents without terms in ngram_lst are not given points.
		
	rank_sents_frm_top_ranked_docs()
		
		1. combine_ngrams(): generate a set of top ngrams, e.g, given 2 top ngrams 'hurricane harvey victims' and 'federal emergency management agency said', we get 
		   [
		   	{'hurricane', 'harvey', 'victims'},
		   	{'federal', 'emergency', 'management', 'agency', 'said'}
		   ]
		
		rank_sents_frm_top_ranked_docs()
		2. For all top ranked documents (from get_ranked_docs()), 

			  get_docs_sentence_score():
		      - For all sentences in a top ranked doc, assign a sentence score (average overlap) by measuring overlap between all the top ngrams in 1. and a given sentence (calc_avg_overlap()). This account for how many different tokens in the top ngrams that a sentence has.

		3. Sentences are subsequently ranked according to their respective average overlap scores (highest - best, lowest - worst)

	Sample output
		Harvey small
			 rank  sumgram                                              TF   TF-Rate
			  1    hurricane harvey                                     18    0.90 
			  2    the federal emergency management agency              8     0.40 
			  3    a category 4 hurricane                               7     0.35 
			  4    corpus christi                                       7     0.35 
			  5    the gulf coast                                       7     0.35 
			  6    president trump                                      7     0.35 
			  7    flooded homes                                        6     0.30 
			  8    tropical storm harvey                                6     0.30 
			  9    the agency said                                      5     0.25 
			  10   the george r. brown convention center                5     0.25 
			  11   the houston area                                     5     0.25 
			  12   hurricane irma                                       5     0.25 
			  13   last week                                            5     0.25 
			  14   army national guard                                  5     0.25 
			  15   in port aransas                                      5     0.25 
			  16   the red cross                                        5     0.25 
			  17   aftermath hurricane                                  4     0.20 
			  18   aug 25,                                              4     0.20 
			  19   the coastal bend                                     4     0.20 
			  20   courtney sacco/caller-times                          4     0.20 

		Harvey
			 rank  sumgram                                              TF   TF-Rate
			  1    hurricane harvey                                    225    0.50 
			  2    tropical storm harvey                               121    0.27 
			  3    corpus christi                                      116    0.26 
			  4    the national hurricane center                        67    0.15 
			  5    as a category 4 hurricane                            63    0.14 
			  6    the federal emergency management agency              63    0.14 
			  7    the national weather service                         58    0.13 
			  8    port aransas                                         57    0.13 
			  9    the gulf of mexico                                   56    0.13 
			  10   the texas gulf coast                                 53    0.12 
			  11   harvey landfall                                      52    0.12 
			  12   the united states                                    52    0.12 
			  13   inches rain                                          51    0.11 
			  14   storm surge                                          49    0.11 
			  15   a tropical depression                                46    0.10 
			  16   the coastal bend                                     43    0.10 
			  17   tropical cyclone                                     43    0.10 
			  18   the houston area                                     40    0.09 
			  19   harris county                                        38    0.09 
			  20   southeast texas                                      38    0.09 

		Ebola
			 rank  sumgram                                              TF   TF-Rate
			  1    ebola virus                                         224    0.39 
			  2    in west africa                                      147    0.25 
			  3    public health                                       117    0.20 
			  4    sierra leone                                        116    0.20 
			  5    ebola outbreak                                      111    0.19 
			  6    the world health organization                        93    0.16 
			  7    the united states                                    92    0.16 
			  8    centers for disease control and prevention           85    0.15 
			  9    infectious diseases                                  81    0.14 
			  10   health care workers                                  63    0.11 
			  11   democratic republic of the congo                     58    0.10 
			  12   bodily fluids                                        57    0.10 
			  13   ebola hemorrhagic fever                              55    0.09 
			  14   direct contact with                                  54    0.09 
			  15   21 days                                              51    0.09 
			  16   outbreak west                                        48    0.08 
			  17   outbreak ebola                                       47    0.08 
			  18   disease evd                                          43    0.07 
			  19   guinea liberia                                       42    0.07 
			  20   body fluids                                          41    0.07 



