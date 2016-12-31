"""
Entry point for triple scoring task.

Example calls: 

	relsifter -i profession.train -o ../output/

	python __main__.py -i profession.train -o ../output/

	python __main__.py -i nationality.train -o ../output/
"""
import sys
import os
import csv
import argparse
import numpy as np
import pandas as pd
import cPickle as pkl
import urllib

from pandas import DataFrame, Series
from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from time import time
from datetime import datetime
from multiprocessing import cpu_count
from scipy.sparse import hstack
from numpy.random import multinomial

from relsifter import LOG, Logger, DATAMAP, GRAPH_PATH, SHAPE, CLIP
from relsifter.activity.feature_extraction import FeatureExtractor
from relsifter.textprofile.feature_extraction import AbstractFeatureExtractor

ncpu = cpu_count()
_min, _max = 2, 5
scale = np.arange(_min, _max + 1)

# distribution for random predictions: based on a histogram in the paper.
pvals = np.array([.13,.11,.12,.08,.10,.14,.10,.26]) 
pvals = pvals / pvals.sum()
pvals[_min] = pvals[:_min+1].sum()
pvals[_max] = pvals[_max:].sum()
pvals = pvals[_min:_max+1]

def get_random_prediction(size=1):
	val = np.nonzero(multinomial(1, pvals, size))[1][:size] + _min
	if size == 1:
		val = val[0]
	return val

class RelSifter:
	"""A class RelSifter for feature extraction and prediction of triples."""
	def __init__(self, rel, display=False):
		print "=> Initializing RelSifter for relation '%s' .." % rel
		self.relation = rel
		# read KB of matching triples, based on the input file.
		self.kb = self.read_knowledge_base(display=display)
		
		# load model/classifier(s)
		self.clf = self.load_classifier(display=display)

		# load feature extractors
		self.fex = self.load_feature_extractors(display=display)

	def get_data(self, data):
		if 'profession' in self.relation:
			_map = DATAMAP['profession']
		elif 'nationality' in self.relation:
			_map = DATAMAP['nationality']
		else:
			raise Exception('Relation for RelSifter not initialized.')
		return _map.get(data)

	def read_knowledge_base(self, display=False):
		"""
		Reads the knowledge base corresponding to the target relation.
		e.g. if infile is profession.train or profession.test, the corresponding
		KB file (profession_kb_match.pkl) is read.

		Returns a dictionary of (cup_sub, cup_obj) -> d, dict of information pairs.

		"""
		kb_fname = self.get_data('kb')
		try:
			kb_dict = pkl.load(open(kb_fname, 'rb'))
			if display:
				print '  * KB read success: {}, {}'.format(basename(kb_fname), len(kb_dict))
		except IOError, e:
			raise e
		return kb_dict

	def load_feature_extractors(self, display=False):
		fex = dict()
		kg_features_path = self.get_data('kg_features')
		kg_top_k = self.get_data('top_k')
		if display:
			print '  * Loading KG-based feature extractor..'
		fex['kg'] = FeatureExtractor(
			GRAPH_PATH, SHAPE, kg_features_path, self.relation, kg_top_k
		)
		return fex

	def load_classifier(self, display=False):
		"Simply loads the classifier at the given location."
		if 'profession' in self.relation:
			clf_names = {c:self.get_data(c) for c in ['clf']}
		elif 'nation' in self.relation:
			clf_names = {c:self.get_data(c) for c in ['clf']}
		else:
			raise Exception('Unrecognized target relation: %s' % rel)
		try:
			clf = dict()
			for c, cname in clf_names.iteritems():
				with open(cname, 'rb') as g:
					_clf = pkl.load(g)
					clf[c] = _clf.get('clf')
					if display:
						print '  * Read classifier: {}'.format(basename(cname))
		except IOError, e:
			raise e
		return clf
		
	def deduplicate_triples(self, infile, delim='\t', display=False):
		"""
		Identifies the set of match and no-match triples in our KB.
		Returns a list of triples (same order as input file) where each item
		is a dictionary having the following as keys.
		* found: True/False
		* cup_sub: input sub name
		* cup_obj: input obj name
		* sub: matching sub name in KB
		* obj: matching obj name in KB
		* pred: predicate in KB
		* sid: id of sub in KB
		* oid: id of obj in KB
		* pid: id of pred in KB

		Note: Only (cup_sub, cup_obj, found) are available for triples that do not
		match with KB.
		"""
		# read input file.
		try:
			unquote = lambda k: urllib.unquote(k)
			df = pd.read_table(
				infile, sep=delim, header=None, quoting=csv.QUOTE_NONE, 
				usecols=[0,1], converters={0: unquote, 1:unquote}
			)
			ntriples = df.shape[0]
			df = df.iloc[:, :2]
			df.columns = ['subs', 'objs']
			d = df.to_dict(orient='list')
			if display:
				print '=> Read input file: {} {}'.format(basename(infile), df.shape)
		except Exception, e:
			raise e

		triples = []
		match = 0
		for idx, item in enumerate(zip(d['subs'], d['objs'])):
			if item in self.kb:
				match += 1
				t = dict(self.kb[item].items() + [('found', True)])
			else:
				t = {'found': False}
			t['cup_sub'] = item[0]
			t['cup_obj'] = item[1]
			triples.append(t)
		if display:
			print '  * #Matching triples (KB/INPUT): {}/{}'.format(match, ntriples)
		return triples

	def predict_triple(self, triple):
		"Extracts features and makes prediction for a single triple."
		# extract features for this triple.
		t1 = time()
		fvec = self.fex['kg'].features_for_triple(triple)
		fex_time = time() - t1
		t1 = time()
		score = self.clf['clf'].predict(fvec)[0] # make a prediction
		pred_time = time() - t1
		return score, fex_time, pred_time

	def predict_triples(self, triples, clip=True):
		scored_triples = []
		# first let's extract all features
		from scipy.sparse import vstack, csr_matrix
		features = []
		t1 = time()
		found = np.zeros(len(triples), dtype=np.int64)
		print 'Extracting features..'
		step, progress = 0, np.round(np.linspace(0, len(triples), 5)[1:]) - 1
		for idx, triple in enumerate(triples):
			if idx == progress[step]:
				step += 1
				print '{}% complete.'.format(25 * (1 + np.where(progress == idx)[0][0]))
			if triple.get('found') is None or not triple['found']:
				found[idx] = 0
				continue
			found[idx] = 1
			fvec = csr_matrix(self.fex['kg'].features_for_triple(triple))
			features.append(fvec)
		features = vstack(features)
		print 'Feature extraction complete: {:.5f}s'.format(time() - t1)
		sys.stdout.flush()

		# prediction phase
		print 'Making predictions..',
		t1 = time()
		not_found = len(np.where(found == 0)[0]) # number of not found triples
		found_preds = np.ones(features.shape[0]) * -1
		step = 0
		progress = np.round(np.linspace(0, features.shape[0], 50)[1:]) - 1
		for step in xrange(1 + len(progress)):
			start = int(progress[step-1]) if step <> 0 else 0
			end = int(progress[step]) if step < len(progress) else features.shape[0]
			found_preds[start:end] = self.clf['clf'].predict(features[start:end,:])
		print 'complete'
		all_preds = np.zeros(len(triples), dtype=np.int64)
		all_preds[np.where(found == 1)] = found_preds
		all_preds[np.where(found == 0)] = get_random_prediction(not_found)
		# clip scores
		if clip:
			all_preds[np.in1d(all_preds, np.array([6,7]))] = 5
			all_preds[np.in1d(all_preds, np.array([0,1]))] = 2
		print 'Prediction complete: {:.5f}s'.format(time() - t1)
		sys.stdout.flush()

		# update list of triples with predictions
		for idx, triple in enumerate(triples):
			triple['score'] = all_preds[idx]
			scored_triples.append(triple)
		return scored_triples


	def _predict_triples(self, triples):
		"""
		Performs triple scoring prediction for the list of input triples.
		Input: 
			- list of triples, where each triple is a dict containing 
				'sid', 'oid' and 'found' in minimum.
		Output: Returns the same list with a score key-value added to the dict.
		"""
		scored_triples = []
		fex_time = np.ones(len(triples)) * -1
		pred_time = np.ones(len(triples)) * -1
		for idx, triple in enumerate(triples):
			if triple.get('found') is not None and triple['found']:
				score, ftime, ptime = self.predict_triple(triple)
				fex_time[idx] = ftime
				pred_time[idx] = ptime
			else:
				score = get_random_prediction() # make a random prediction
			# clip score
			if score in [0, 1]:
				score = 2
			elif score in [6, 7]:
				score = 5
			triple['score'] = score
			scored_triples.append(triple)
		print 'Avg. feature extraction time: {:.5f}s'.format(fex_time[np.where(fex_time != -1)].mean())
		print 'Avg. prediction time: {:.5f}s'.format(pred_time[np.where(pred_time != -1)].mean())
		return scored_triples

def triple_scoring(args, display=False):
	"Entry point for dealing with input triple files."
	for infile in args.infiles:
		if 'profession' in infile:
			relation = 'profession'
		elif 'nation' in infile:
			relation = 'nationality'
		rsifter = RelSifter(relation, display=display)
		
		# deduplicate
		triples = rsifter.deduplicate_triples(infile, display=display) # dict

		# predict for each triple in the list
		t1 = time()
		print '=> Feature extraction and predictions ...'
		scored_triples = rsifter.predict_triples(triples, clip=CLIP)
		print '=> Scoring time: {:.2f} secs.'.format(time() - t1)

		# clear all data structures
		rsifter.fex['kg'].G = None
		rsifter.kb = None
		rsifter.clf = None
		
		# output to the dir, after appropriate format transformations
		pred = DataFrame.from_records(scored_triples)
		pred = pred[['cup_sub', 'cup_obj', 'score']]
		outfile = join(args.outdir, basename(infile))
		pred.to_csv(outfile, sep='\t', header=False, index=False, \
			quoting=csv.QUOTE_NONE)
		print '=> Saved output: {}\n'.format(outfile)

def main(args=None):
	# parse arguments
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-i', type=str, nargs='+', required=True, action='append',
			dest='infiles', help='Input files containing pairs from type-like relations.')
	parser.add_argument('-o', type=str, required=True,
			dest='outdir', help='Path to the output directory.')
	args = parser.parse_args()

	# set up log file first
	_now = datetime.now()
	log = join(LOG, 'relsifter-{}.log'.format(_now.strftime("%Y-%m-%d-%H-%M-%S")))
	sys.stdout = Logger(log)
	print '[Relsifter launched]: {}'.format(_now.strftime("%Y-%m-%d %H:%M:%S"))
	print '=> Log file: {}'.format(log)
	print ''

	# ensure input files and output directory is valid.
	outdir = abspath(expanduser(args.outdir))
	if not exists(outdir):
		raise Exception('Output directory does not exist: %s' % outdir)
	if not isdir(outdir):
		raise Exception('Not a directory: %s' % outdir)
	args.outdir = outdir
	infiles = []
	args.infiles = [i[0] for i in args.infiles]
	for infile in args.infiles:
		infile = abspath(expanduser(infile))
		if not exists(infile):
			raise Exception('File %s does not exist.' % infile)
		if not isfile(infile):
			raise Exception('Not a file: %s' % infile)
		infiles.append(infile)	
	args.infiles = infiles

	# show user IO params
	print '=> Input files:'
	for i, f in enumerate(args.infiles):
		print '  [{}] {}'.format(i+1, f)
	print '=> Output directory: {}'.format(args.outdir)
	print ''

	# Perform triple scoring
	print 'Performing triple scoring..'
	ts = time()
	triple_scoring(args, display=True)
	print 'Total time taken: {:.2f} secs.'.format(time() - ts)
	print '\n[Relsifter complete]: {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	sys.stdout.flush()
	print ''

if __name__ == '__main__':
	main()