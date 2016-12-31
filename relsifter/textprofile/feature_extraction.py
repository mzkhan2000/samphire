"""
Extracts Wikipedia articles' abstract-based features for a set of triples
specified in an input file. However, the code is generic enough to handle
any kind of free-form text besides abstract.

** Note: target_dedup and person_dedup are needed as arguments because the abstracts
are not de-duplicated (i.e. redirects may not be resolved). So the match files help to
find DBpedia entities for entities given by the cup or Wikipedia articles.

"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import shutil
import re
import unicodedata
import cPickle as pkl

from time import time
from pandas import DataFrame, Series, merge
from datetime import date
from os.path import abspath, expanduser, dirname, isfile, isdir, exists,\
	join, basename, splitext
from scipy.sparse import hstack

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

TABLE = dict.fromkeys(i for i in xrange(sys.maxunicode) \
			if unicodedata.category(unichr(i)).startswith('P'))


class LemmaTokenizer(object):
	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.table = TABLE
		
	def __call__(self, doc):
		doc = doc.translate(self.table)
		return [self.lemmatizer.lemmatize(t) for t in word_tokenize(doc)]

class AbstractFeatureExtractor:
	"""Feature extractor for Wikipedia text/abstract of individuals in triples."""
	def __init__(self, rel_abs_fname, person_abs_fname, all_triples_fname, load=True, top_k=5):
		if load:
			return
		self.top_k = top_k
		self.rel_abs_fname = rel_abs_fname
		self.person_abs_fname = person_abs_fname
		self.all_triples_fname = all_triples_fname
		rel = None
		if 'profession' in self.rel_abs_fname:
			rel = 'profession'
		elif 'nation' in self.rel_abs_fname:
			rel = 'nationality'
		else:
			raise Exception('Unrecognized target relation %s' % self.rel_abs_fname)
		self.fex_obj_fname = 'fex_abs_{}_features_top{}.pkl'.format(rel, self.top_k)

		# read abstract as a dict (entity, abstract) pairs
		rel_abs = self.read_abstract(rel_abs_fname)

		# learn vocab, top-k features
		vectorizer = self.extract_features(rel_abs.items(), top_k)
		# read person-abstract pairs as dict
		person_abs = self.get_person_abstract(person_abs_fname, vectorizer)
		# print person_abs
		# read all triples in cup
		triples, self.triples_idx = self.read_cup_triples(all_triples_fname)
		# learn DictVectorizer (feature-matrix object)
		self.dvec = self.learn_feature_matrix(triples, person_abs, vectorizer)
		# save the object
		self.save_feature_extractor(dirname(self.rel_abs_fname))

	def features_for_triple(self, triple):
		"Returns a feature vector for the given triple."
		return self.get_feature_matrix([triple])

	def get_feature_matrix(self, triples):
		"Returns a feature matrix for the list of triples."
		idx = [self.triples_idx[triple] for triple in triples]
		X_abs = self.dvec.dvec_mat[idx, :]
		return X_abs

	def save_feature_extractor(self, outdir):
		"""Saves bare-minimum AbstractFeatureExtractor object, including 
		* triples_idx: dict of (sub, obj) -> index into the DictVectorizer
		* dvec: DictVectorizer representing the feature matrix of all triples in the cup.
		"""
		fname = join(outdir, self.fex_obj_fname)
		try:
			d = {'triples_idx': self.triples_idx, 'dvec': self.dvec}
			with open(fname, 'wb') as g:
				pkl.dump(d, g, protocol=pkl.HIGHEST_PROTOCOL)
			print 'Saved AbstractFeatureExtractor object: {}'.format(fname)
		except IOError, e:
			print 'Error saving AbstractFeatureExtractor object.'
			raise e

	def load_feature_extractor(self, fname):
		try:
			if not exists(fname):
				raise Exception('AbstractFeatureExtractor file %s not exists' % fname)
			with open(fname, 'rb') as g:
				d = pkl.load(g)
			self.triples_idx = d.get('triples_idx')
			self.dvec = d.get('dvec')
		except Exception, e:
			raise e

	def learn_feature_matrix(self, triples, person_abs, vectorizer):
		"""
		Learns a sparse DictVectorizer object representing a feature matrix
		that can be used in building machine learning models.

		Runs through all the triples, looking up their abstracts in provided 
		abstract cache (person_abs) and creates feature dictionary for each triple.
		Feature dictionaries from all triples in the cup are then used to create a
		DictVectorizer object, which represents the abstract-based feature matrix.
		"""
		def _get_feature_dict(triple):
			sub, obj = triple
			obj_idx = vectorizer.target_idx_cache.get(obj)
			if obj_idx is None or sub not in person_abs:
				return dict() # no tokens/abstract available for this subject. return empty dictionary
			abs_tokens = person_abs[sub]
			d = {}
			for token in abs_tokens:
				if token in vectorizer.top_feature_idx and token not in d:
					d[token] = vectorizer.td_mat[obj_idx, vectorizer.top_feature_idx[token]]
			return d	
		print 'Creating feature dictionaries..',
		sys.stdout.flush()
		# create list of feature dictionaries, one for each triple
		D = []
		t1 = time()
		person_no_features = 0
		for i, triple in enumerate(triples):
			d = _get_feature_dict(triple)
			if len(d) == 0:
				person_no_features += 1
			D.append(d)
		print '#People w/o features: {}. Time: {:.2f}s'.format(person_no_features, time() - t1)

		# create a sparse DictVectorizer object, represeting the feature matrix
		dvec = DictVectorizer(sparse=True)
		dvec.feature_names_ = vectorizer.top_feature_idx.keys()
		dvec.dvec_mat = dvec.fit_transform(D)
		return dvec

	def read_cup_triples(self, all_triples_fname):
		"""Reads all the triples provided by the cup. Returns list of triples 
		and a dict of (triple, index in the list) pairs.
		"""
		rel_all = pd.read_table(all_triples_fname, sep=',', header=0)
		rel_all = rel_all[['sub', 'obj']].to_dict(orient='list')
		triples = zip(rel_all['sub'], rel_all['obj'])
		triples_idx = {t:idx for idx, t in enumerate(triples)}
		return triples, triples_idx

	def read_abstract(self, fname):
		"Returns a dict of Wikipedia entity -> abstract pairs."
		abs_cache = dict()
		with open(fname) as g:
			for line in g:
				line = line.decode('UTF-8')
				tokens = line.strip().split(' ')
				sub, obj = tokens[0], ' '.join(tokens[1:])
				abs_cache[sub] = obj
		return abs_cache

	def get_person_abstract(self, person_abs_fname, vectorizer):
		"Reads all the abstracts associated with persons given in the Cup data."
		tkns_abs_dir, tkns_abs_fname = dirname(person_abs_fname), basename(person_abs_fname)
		tkns_person_abs_fname = join(tkns_abs_dir, 'tokenized_{}'.format(tkns_abs_fname))
		if exists(tkns_person_abs_fname):
			print 'Reading person abstracts..',
			sys.stdout.flush()
			person_abs = self.read_abstract(tkns_person_abs_fname).items()
			person_abs = dict([(k, v.split(' ')) for k, v in person_abs])
			print 'Person abstracts read complete.'
		else:
			person_abs = self.read_abstract(person_abs_fname).items()
			print 'Creating tokenized abstracts..'
			sys.stdout.flush()
			_analyzer = vectorizer.build_analyzer()
			if not exists(tkns_person_abs_fname):
				t1 = time()
				with open(tkns_person_abs_fname, 'w') as g:
					for i, (k, abstract) in enumerate(person_abs):
						abstract = ' '.join(_analyzer(abstract))
						k, abstract = (t.encode('UTF-8') for t in (k, abstract))
						g.write('{} {}\n'.format(k, abstract))
				print 'Saved: {}'.format(tkns_person_abs_fname)
				print 'Abstracts (stopwords removed, stemmed & lemmatized): {:.2f}s'.format(time() - t1)    
		return person_abs

	def extract_features(self, cache, top, display=False):
		"""
		Learns a vocabulary based on abstracts and then extracts top k words for each 
		target profession/nationality.
		* cache: a list of (name, abstract) tuples for nationality/profession.
		* top: Number of top ranked words to use for profession/nationality
		"""
		# Learn vocabulary by preprocessing, tokenizing, lemmatization and performing TFIDF
		docs = [doc for _, doc in cache]
		vectorizer = TfidfVectorizer(
			encoding='utf-8', strip_accents='ascii', lowercase=True, 
			preprocessor=None, tokenizer=LemmaTokenizer(), analyzer=u'word', 
			stop_words='english', token_pattern=u'(?u)\b\w\w+\b', max_df=1.0, min_df=1, 
			max_features=None, norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False
		)
		td_mat = vectorizer.fit_transform(docs)
		analyzer = vectorizer.build_analyzer()
		vocab = vectorizer.get_feature_names()
		vectorizer.td_mat = td_mat
		vectorizer.inv_vocabulary_ = {idx: tkn for tkn, idx in vectorizer.vocabulary_.iteritems()} # featureidx -> feature
		vectorizer.idx_target_cache = {idx:k for idx, (k, _) in enumerate(cache)} # rowidx -> profession/nationality
		vectorizer.target_idx_cache = {v:k for k, v in vectorizer.idx_target_cache.iteritems()} # prof/nat -> rowidx
		print 'Vocabulary length: {}'.format(len(vocab))

		# Extract top-k features
		indices = np.arange(td_mat.shape[0])
		top_feature_idx = map(lambda rowidx: set(np.argsort(td_mat[rowidx,:].toarray()[0])[::-1][:top]), indices)
		if display:
			features_dir, fn = dirname(self.rel_abs_fname), splitext(basename(self.rel_abs_fname))[0]
			abs_features_fname = join(features_dir, 'top_{}_features_{}.txt'.format(fn, top))
			with open(abs_features_fname, 'w') as g:
				for target in indices:
					print '=> {}'.format(vectorizer.idx_target_cache[target])
					feat_tfidf = [(vectorizer.inv_vocabulary_[t], td_mat[target, t]) for t in top_feature_idx[target]]
					feat_tfidf = sorted(feat_tfidf, key=lambda x: x[1], reverse=True)
					ff = ''
					for feat, tfidf in feat_tfidf:
						s = '[{} {:.2f}] '.format(feat, tfidf)
						ff += ' ' + s
						print s,
					g.write('{} {}\n'.format(vectorizer.idx_target_cache[target], ff))
					print '\n'
				print 'Saved features: {}'.format(abs_features_fname)
		top_feature_idx = reduce(lambda a, b: a | b, top_feature_idx)
		vectorizer.top_feature_idx = {vectorizer.inv_vocabulary_[idx]:idx for idx in top_feature_idx} # featurename -> idx
		print '#Features associated w/ top {} words: {}'.format(top, len(top_feature_idx))
		return vectorizer

if __name__ == '__main__':
	"""
	Example call:

	# profession
	python feature_extraction.py 
		-target_abstract ../../data/processed/abstracts/professions_abstract.txt 
		-path ../../data/processed/cup/profession_kb_match.csv 
		-person_abstract ../../data/processed/abstracts/persons_abstract.txt 
		-top 5
 
 	# nationality
	python feature_extraction.py
		-target_abstract ../../data/processed/abstracts/nationality_abstract.txt
		-path ../../data/processed/cup/nationality_kb_match.csv
		-person_abstract ../../data/processed/abstracts/persons_abstract.txt 
		-top 5

	"""
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-target_abstract', type=str, required=True, dest='target_abstract', 
		help='Path to file containing abstracts of professions/nationalities.')
	parser.add_argument('-path', type=str, required=True,
		dest='path', help='Path to (training) triples file.')
	parser.add_argument('-person_abstract', type=str, required=True, dest='person_abstract',\
		help='Path to file containing abstracts of persons.')
	parser.add_argument('-outpath', type=str, 
			dest='outpath', help='Absolute path to the output directory.')
	parser.add_argument('-top', type=int, nargs='+', dest='topfeatures', \
			default=5, help='Top k features to use.')
	args = parser.parse_args()

	args.path = abspath(expanduser(args.path))
	args.target_abstract = abspath(expanduser(args.target_abstract))
	args.person_abstract = abspath(expanduser(args.person_abstract))
	for fname in [args.target_abstract, args.path, args.person_abstract]:
		if not exists(fname):
			raise Exception('File not found: %s' % fname)
		if not isfile(fname):
			raise ValueError('Input is not a file: %s' % fname)
	if args.outpath is not None:
		args.outpath = abspath(expanduser(args.outpath)) 
	else:
		args.outpath = dirname(args.target_abstract)
	
	if isinstance(args.topfeatures, int):
		args.topfeatures = [args.topfeatures]
	rel = None
	if 'profession' in args.target_abstract:
		rel = 'profession'
	elif 'nation' in args.target_abstract:
		rel = 'nationality'
	else:
		raise Exception('Unrecognized relation type (profession/nationality).')

	# feature extraction for every value of top_k
	for top in args.topfeatures:
		# Feature extractor object
		t1 = time()
		fex = AbstractFeatureExtractor(
			args.target_abstract, args.person_abstract, args.path, top_k=top, load=False
		)
		print 'Time taken for feature extraction: {:.2f} secs.'.format(time() - t1)
	print '\nDone!\n'
