import json
import unittest

from sumgram.sumgram import get_top_sumgrams

class TestSumgram(unittest.TestCase):
    
    ngram = 2
    def test_basic_script(self):

        doc_lst = [
            {'id': 0, 'text': 'The eye of Category 4 Hurricane Harvey is now over Aransas Bay. A station at Aransas Pass run by the Texas Coastal Observing Network recently reported a sustained wind of 102 mph with a gust to 132 mph. A station at Aransas Wildlife Refuge run by the Texas Coastal Observing Network recently reported a sustained wind of 75 mph with a gust to 99 mph. A station at Rockport reported a pressure of 945 mb on the western side of the eye.'},
            {'id': 1, 'text': 'Eye of Category 4 Hurricane Harvey is almost onshore. A station at Aransas Pass run by the Texas Coastal Observing Network recently reported a sustained wind of 102 mph with a gust to 120 mph.'},
            {'id': 2, 'text': 'Hurricane Harvey has become a Category 4 storm with maximum sustained winds of 130 mph. Sustained hurricane-force winds are spreading onto the middle Texas coast.'}
        ]
        params = {'top_sumgram_count': 10}
        sumgrams = get_top_sumgrams(doc_lst, TestSumgram.ngram, params=params)

        self.assertTrue( sumgrams['top_sumgrams'][0]['ngram'] != '', "Error statement: sumgrams['top_sumgrams'][0]['ngram']" )
        self.assertGreater( sumgrams['ranked_docs'][0]['score'], 0, "sumgrams['ranked_docs'][0]['score']" )
        self.assertGreater( sumgrams['ranked_sentences'][0]['avg_overlap'], 0, "sumgrams['ranked_sentences'][0]['avg_overlap']" )
    
    def test_multiple_opts(self):

        doc_lst = [
            {'id': 0, 'text': 'The eye of Category 4 Hurricane Harvey is now over Aransas Bay. A station at Aransas Pass run by the Texas Coastal Observing Network recently reported a sustained wind of 102 mph with a gust to 132 mph. A station at Aransas Wildlife Refuge run by the Texas Coastal Observing Network recently reported a sustained wind of 75 mph with a gust to 99 mph. A station at Rockport reported a pressure of 945 mb on the western side of the eye.'},
            {'id': 1, 'text': 'Eye of Category 4 Hurricane Harvey is almost onshore. A station at Aransas Pass run by the Texas Coastal Observing Network recently reported a sustained wind of 102 mph with a gust to 120 mph.'},
            {'id': 2, 'text': 'Hurricane Harvey has become a Category 4 storm with maximum sustained winds of 130 mph. Sustained hurricane-force winds are spreading onto the middle Texas coast.'}
        ]
        params = {
            'top_sumgram_count': 10,
            'add_stopwords': ['image'],
            'no_rank_docs': True,
            'no_rank_sentences': True,
            'title': 'Top sumgrams for Hurricane Harvey text collection'
        }
        sumgrams = get_top_sumgrams(doc_lst, TestSumgram.ngram, params=params)
        
        self.assertTrue( sumgrams['top_sumgrams'][0]['ngram'] != '', "Error statement: sumgrams['top_sumgrams'][0]['ngram']" )
        self.assertTrue( 'ranked_docs' not in sumgrams , "'ranked_docs' not in sumgrams" )
        self.assertTrue( 'ranked_sentences' not in sumgrams , "'ranked_sentences' not in sumgrams" )


if __name__ == '__main__':
    unittest.main()