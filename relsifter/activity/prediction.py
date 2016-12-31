"""
Extracts features and makes predictions for type-like triples
using previously build machine learned models.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import cPickle as pkl

from pandas import DataFrame, Series
from datetime import date
from os.path import abspath, expanduser, dirname, isfile, isdir, exists,\
	join, basename, splitext
from time import time
from multiprocessing import Pool, cpu_count
from contextlib import closing
from scipy.sparse import csr_matrix, vstack

# from feature_extraction import FeatureExtractor

_min, _max = 2, 5
scale = np.arange(_min, _max + 1)

MAX_TASK_PER_WORKER = 1000
WORKER_DATA = dict()

def _init_worker(clf, fex):
	global WORKER_DATA
	WORKER_DATA['clf'] = clf
	WORKER_DATA['fex'] = fex
	
def _worker(triple):
	global WORKER_DATA
	clf = WORKER_DATA['clf']
	fex = WORKER_DATA['fex']
	if triple.get('found') is not None and triple['found']:
		# make a prediction using the given classifier.
		fvec = fex.features_for_triple(triple)
		score = clf.predict(fvec)[0]
	else:
		# make random prediction between 0 through 7.
		score = np.random.choice(scale)
	# clip the score
	if score in [0, 1]:
		score = 2
	elif score in [6, 7]:
		score = 5
	triple['score'] = score
	return triple

def predict_triples_fast(triples, clf, fex):
	ncpu = cpu_count()
	nprocs = min(2, ncpu)
	print 'Launching pool of {} workers among {} CPUs.'.format(nprocs, ncpu)
	chunksize = len(triples) // nprocs
	print 'Chunk size for each process: {}'.format(chunksize)
	sys.stdout.flush() # flush all output
	pool = Pool(processes=nprocs, initializer=_init_worker,
				initargs=(clf, fex), maxtasksperchild=MAX_TASK_PER_WORKER)
	try:
		with closing(pool):
			results = pool.map_async(_worker, triples, chunksize=chunksize)
			while not results.ready():
				results.wait(1)
		pool.join()
		if results.successful():
			scored_triples = results.get()
		else:
			err = results.get()
			print >> sys.stderr, "There was an error in the pool: {}".format(err)
			sys.exit(2)  # ERROR occurred
	except KeyboardInterrupt:
		print "^C"
		pool.terminate()
		sys.exit(1)  # SIGINT received
	except Exception, e:
		print e	
	return scored_triples

def predict_triples(triples, clf, fex):
	"""
	Performs triple scoring prediction for the list of input triples.
	Input: 
		- list of triples, where each triple is a dict containing 
			'sid', 'oid' and 'found' in minimum.
		- previously built machine learning model (clf).
		- fex: FeatureExtractor object in feature_extraction module.
		- k: Number of top features to extract.
	Output: Returns the same list with a score key-value added to the dict.
	"""
	scored_triples = []
	for idx, triple in enumerate(triples):
		if triple.get('found') is not None and triple['found']:
			# extract features for this triple.
			fvec = fex.features_for_triple(triple)

			# make a prediction using the given classifier.
			score = clf.predict(fvec)[0]
		else:
			# make random prediction between 0 through 7.
			score = np.random.choice(scale)
		# clip the score.
		if score in [0, 1]:
			score = 2
		elif score in [6, 7]:
			score = 5
		triple['score'] = score
		scored_triples.append(triple)
	return scored_triples

if __name__ == '__main__':
	"""
	Example call:

	python prediction.py 
		-test ../../data/processed/cup/profession_train.csv
		-clf ../../data/processed/cup/experiment5/clf_randomForest_profession_train_features_top5.pkl
		-g ../../data/processed/kg/_undir
		-shape 6060993 6060993 663
		-feat ../../data/processed/cup/experiment5/professions
		
	"""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-test', type=str, dest='test', required=True, \
		help='Test file path.')
	parser.add_argument('-clf', type=str, dest='clf', required=True, \
		help='Path to the pickled classifier file.')
	parser.add_argument('-g', type=str, required=True,
		dest='graphbase', help='Directory where graph vectors are located.')
	parser.add_argument('-shape', type=int, nargs='+', required=True, 
		dest='shape', help='Graph shape.')
	parser.add_argument('-feat', type=str, required=True, dest='feat', 
		help='Directory where professions/nationalities feature files are located.')
	parser.add_argument('-top', type=int, dest='topfeatures', \
			default=5, help='Top k features to use.')
	parser.add_argument('-D', '--outdir', \
		help='Output directory where to save the predictions.')
	args = parser.parse_args()
	print 

	# path related checks.
	args.graphbase = abspath(expanduser(args.graphbase))	
	args.feat = abspath(expanduser(args.feat))
	test = abspath(expanduser(args.test))
	if not exists(test) or not isfile(test):
		raise Exception('Not a test file or does not exist: %s' % test)
	args.test = test
	clf = abspath(expanduser(args.clf))
	if not exists(clf) or not isfile(clf) or splitext(clf)[1] <> '.pkl':
		raise Exception('Classifier does not exist, \
			or is not a (pickled) file: %s' % clf)
	args.clf = clf
	if args.outdir is None:
		args.outdir = dirname(args.test)
	outdir = abspath(expanduser(args.outdir))
	if not exists(outdir) or not isdir(outdir):
		raise Exception('Not a directory or does not exist: %s' % outdir)
	args.outdir = outdir

	# read test file
	df = pd.read_table(args.test, sep=',', header=0)
	df = df.dropna() # only valid triples: w/ sid, oid are considered.
	df['found'] = True 
	triples = df.to_dict(orient='records')
	print 'Read test data: {} {}'.format(basename(args.test), len(triples))
	
	rel = None
	if 'profession' in args.test:
		rel = 'profession'
	elif 'nation' in args.test:
		rel = 'nationality'
	else:
		raise Exception('Unrecognized relation type (profession/nationality).')

	# read classifier
	clf = None
	try:
		with open(args.clf, 'rb') as g:
			clf = pkl.load(g)
			clf = clf.get('clf')
			print 'Read classifier: {}'.format(args.clf)
	except IOError, e:
		raise e

	# create feature extractor object
	fex = FeatureExtractor(args.graphbase, args.shape, args.feat, rel, args.topfeatures)

	# make predictions
	t1 = time()
	print 'Making predictions..'
	scored_triples = predict_triples_fast(triples, clf, fex)
	print 'Prediction complete. Time: {:.2f} secs.'.format(time() - t1)

	# save
	pred_df = DataFrame.from_records(scored_triples)
	pred_df = pred_df[list(df.columns) + list(pred_df.columns - set(df.columns))]
	outfile = join(args.outdir, 'predictions_{}.csv'.format(splitext(basename(args.test))[0]))
	pred_df.to_csv(outfile, sep=',', header=True, index=False)
	print 'Saved predictions: {}'.format(outfile)

	print '\nDone!\n'