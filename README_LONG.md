# ngramsum

ngramsum is a tool that
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

Old GetTopNgrams Explanation
	
	get_top_ngrams
		gen_top_ngrams
		merge_split_ngrams
			gen_inc_bridge_ngrams
			get_bridge_ngrams
			merge_inc_ngrams
			update_top_k_ngrams_with_merged
		get_ranked_docs
		rank_sents_frm_top_ranked_docs
		add_removed_words


	gen_top_ngrams()

		generates list of n (e.g., if n = 2, it generates bigrams) ngrams with their respective frequencies and postings:
		[
			{
		        "ngram": "hurricane harvey",
		        "term_freq": 19,
		        "postings": [	
		            {
		                "doc_indx": 0,
		                "id": 0,
		                "details": {}
		            },
		            {
		                "doc_indx": 1,
		                "id": 1,
		                "details": {}
		            }
		        ]
	        },
		    {
		    	...
		    }
	    ]

	gen_inc_bridge_ngrams()
		
		Multiple lower order ngrams (e.g., 'emergency management' and 'federal emergency') may belong to 
		the same higher order ngram (e.g, 'federal emergency management'): responsible for generating a dictionary of (n + 1) ngrams as keys and values are top ngrams with their respective frequencies and postings that are perfect subsets of the (n + 1) ngram key. gen_inc_bridge_ngrams() returns:

		{
			"federal emergency management": 
			[
		        {
		            "top_ngram_pos": 1,
		            "top_ngram": {
		                "ngram": "emergency management",
		                "term_freq": 8,
		                "postings": [
		                    {
		                        "doc_indx": 0,
		                        "id": 0,
		                        "details": {}
		                    },
		                    {
		                        "doc_indx": 1,
		                        "id": 1,
		                        "details": {}
		                    },
		                    {
		                        "doc_indx": 5,
		                        "id": 6,
		                        "details": {}
		                    }
		                ]
		            }
		        },
		        {
		            "top_ngram_pos": 5,
		            "top_ngram": {
		                "ngram": "federal emergency",
		                "term_freq": 7,
		                "postings": [
		                    {
		                        "doc_indx": 0,
		                        "id": 0,
		                        "details": {}
		                    },
		                    {
		                        "doc_indx": 1,
		                        "id": 1,
		                        "details": {}
		                    }
		                ]
		            }
		        }
	    	]
	    }

	get_bridge_ngrams()
		
		Adjacent higher order ngrams (e.g., 'federal emergency management' and 'emergency management agency') may include a common lower order ngram (e.g., 'emergency management'), such adjacent higher order ngrams ought to be merged into a single higher order ngram (e.g., 'federal emergency management agency'). Because the goal of merging split ngrams is to replace a lower order ngram with a single higher order ngram, but if the lower order ngram has multiple parents, we cannot decide which higher order ngram would replace the lower order ngram.

		Therefore get_bridge_ngrams() is responsible for finding adjacent higher order ngrams that include a common lower order ngram

		Given inc_ngram_dct content from gen_inc_bridge_ngrams():
			inc_ngram: federal emergency management
				top_ngram: {'top_ngram_pos': 1, 'top_ngram': ('emergency management', 3, array([0, 1, 2]))}
				top_ngram: {'top_ngram_pos': 6, 'top_ngram': ('federal emergency', 2, array([0, 2]))}

			inc_ngram: emergency management agency
				top_ngram: {'top_ngram_pos': 1, 'top_ngram': ('emergency management', 3, array([0, 1, 2]))}
				top_ngram: {'top_ngram_pos': 2, 'top_ngram': ('management agency', 3, array([0, 1, 2]))}

		get_bridge_ngrams() returns:
			{
				1: ['federal emergency management', 'emergency management agency'], 
				6: ['federal emergency management'], 
				2: ['emergency management agency']
			}

			Gotten by including top_ngram_pos as key and value as the list of the parent inc_ngram,
			e.g., top_ngram: {'top_ngramPos': 1, 'top_ngram': ('emergency management', 3, array([0, 1, 2]))} from inc_ngram: federal emergency management
			yields ['federal emergency management']

			then top_ngram: {'top_ngramPos': 1, 'top_ngram': ('emergency management', 3, array([0, 1, 2]))} emergency management agency
			yields ['federal emergency management', 'emergency management agency']

	merge_inc_ngrams()
		
		Find top_ngrams (called bridge_ngrams) that bridge a pair of inc_ngrams, for such top_ngrams, merge their bridge_ngrams. top_ngrams already in order of most frequent to least frequent.

		Input (bridge_ngrams):
			    "1": [
			        "federal emergency management",
			        "emergency management agency"
			    ],
			    "5": [
			        "federal emergency management"
			    ],
			    "2": [
			        "emergency management agency",
			        "management agency said"
			    ],
			    "14": [
			        "management agency said"
			    ],
			    "0": [
			        "hurricane harvey victims"
			    ],
			    "7": [
			        "hurricane harvey victims"
			    ]
	    
	    Process
	    	top ngram term: "emergency management"
				common inc_ngrams parents: ['federal emergency management', 'emergency management agency']
				
				inc_ngrams parent: 0 'federal emergency management'
				inc_ngrams parent: 1 'emergency management agency'

				merge result: 'federal emergency management agency'
				The top_ngrams (children) of the inc_ngrams will be joined

			top ngram term: 'management agency'
				common inc_ngrams parents: ['emergency management agency', 'management agency said']
				
				inc_ngrams parent: 0 'emergency management agency' (old inc_ngram)
				inc_ngram post update: 'federal emergency management agency' (new inc_ngram)

				inc_ngrams parent: 1 'management agency said'
				merge result: 'federal emergency management agency said' (new inc_ngram + 'management agency said')

		Result
			{
			    "hurricane harvey victims": [
			        {
			            "top_ngram_pos": 0,
			            "top_ngram": {
			                "ngram": "hurricane harvey",
			                "term_freq": 19,
			                "postings": []
			            }
			        },
			        {
			            "top_ngram_pos": 7,
			            "top_ngram": {
			                "ngram": "harvey victims",
			                "term_freq": 7,
			                "postings": []
			            }
			        }
			    ],
			    "federal emergency management agency said": [
			        {
			            "top_ngram_pos": 5,
			            "top_ngram": {
			                "ngram": "federal emergency",
			                "term_freq": 7,
			                "postings": []
			            }
			        },
			        {
			            "top_ngram_pos": 1,
			            "top_ngram": {
			                "ngram": "emergency management",
			                "term_freq": 8,
			                "postings": []
			            }
			        },
			        {
			            "top_ngram_pos": 14,
			            "top_ngram": {
			                "ngram": "agency said",
			                "term_freq": 5,
			                "postings": []
			            }
			        },
			        {
			            "top_ngram_pos": 2,
			            "top_ngram": {
			                "ngram": "management agency",
			                "term_freq": 8,
			                "postings": []
			            }
			        }
			    ]
			}

	update_top_k_ngrams_with_merged()
		
		The top_ngrams (have term frequency) need to be replaced with merged inc_ngrams.
		The merged inc_ngrams (e.g., 'hurricane harvey victims' and 'federal emergency management agency said') are new and derived, and thus do not have term frequency values, and as such need to inherit this value from a single top_ngrams. There are multiple top_ngrams (subset of merged inc_ngrams) possible candidates. So select the top_ngram with the largest frequency.

	get_ranked_docs()
		
		Given i ∈ N = |list of top ngrams|

		Give credit to documents that have highly ranked (bigger diff: N - i) terms in the ngram_lst 
		a document's score is awarded by accumulating the points awarded by the position of terms in the ngram_lst.
		Documenents without terms in ngram_lst are not given points.
		
	rank_sents_frm_top_ranked_docs()
		
		1. combine_ngrams(): generate a set of top ngrams, e.g, given 2 top ngrams 'hurricane harvey victims' and 'federal emergency management agency said', we get 
		   [
		   	{'hurricane', 'harvey', 'victims'},
		   	{'federal', 'emergency', 'management', 'agency', 'said'}
		   ]
		
		rank_sents_frm_top_ranked_docs()
		2. For all top ranked documents (from get_ranked_docs()), 

			  get_docs_sentence_score()
		      For all sentences in a top ranked doc, assign a sentence score (average overlap) by measuring overlap between all the top ngrams in 1. and a given sentence (calc_avg_overlap()). This account for how many different tokens in the top ngrams does a sentence have.

		3. Sentences are subsequently ranked according to their respective average overlap scores (highest - best, lowest - worst)

	add_removed_words()
		
		case 1: Stopwords case removal case:
			The removal of stopword means the top_ngrams have gaps, e.g., 
				"democratic republic congo"
				instead of
				"democratic republic of congo"
		
		case 2: Low occurring terms case:
			Also sometimes grams with lower match are dropped when top ngrams are calculated. For example, "texas" was dropped because it did not occur frequently enough with "gulf coast"
		
		Therefore generated ngram_range ngrams WITH stopwords (called unrestricted ngrams) for a limited set of documents that include lower and higher order ngrams. Next, find top ngrams that are subsets (ensure order is preserved) of the unrestricted ngrams, and select the unrestricted ngram which does not have terms removed (case 1 and case 2)
