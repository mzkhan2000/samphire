"""
Builds the best machine learning model for the Wikipedia abstract training data.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import cPickle as pkl

import mord # ordinal regression
import sklearn

from pandas import DataFrame, Series, merge
from datetime import date
from os.path import abspath, expanduser, dirname, isfile, isdir, exists,\
	join, basename, splitext
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, mean_absolute_error, make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from feature_extraction import AbstractFeatureExtractor

CV = 10
ESTIMATORS = {
	'ordLogReg': {
		'clf': mord.LogisticAT(), 'param': {'alpha': [1, 5, 10, 15, 20, 50, 75, 100, 250, 500, 1000]}
	},
	'randomForest': {
		'clf': RandomForestClassifier(), 'param': {'n_estimators': [10, 50, 100, 250, 500]}
	},
	'adaboost': {
		'clf': AdaBoostClassifier(), 'param': {'n_estimators': [10, 50, 100, 250, 500]}
	},
	'svm': {
		'clf': SVC(), 'param': {'C': [1, 5, 10, 15, 20, 50, 75, 100, 250, 500, 1000]}
	},
	'logistic': {
		'clf': LogisticRegression(), 'param': {'C': [1, 5, 10, 15, 20, 50, 75, 100, 250, 500, 1000]}
	}
}


def get_clipped(ypred):
	ypred = np.asarray(ypred)
	ypred[np.in1d(ypred, [0,1])] = 2    
	ypred[np.in1d(ypred, [6,7])] = 5
	return ypred

# Scoring functions
def acc_fn(y, ypred, k=2, clip=True):
	ypred = np.asarray(ypred)
	y = np.asarray(y)
	if clip:
		ypred = get_clipped(ypred)
	return (np.abs(y - ypred) <= k).sum() / float(len(y))

def omitcols(d, first=7):
	return d.iloc[:, first:]

def find_best_model(X, y, cv=3, preferred_est=None, clip=True, \
					scale=True, skip_est=None, abstract_data=True):
	"""Tries different models on the given data and returns a list of best models (along with their best score) 
	for each parameter setting."""
	# evaluation measure
	acc_scorer = make_scorer(acc_fn, greater_is_better=True, clip=clip)
	
	# make data 
	X_train, X_test, y_train, y_test = train_test_split(X, y, \
				test_size=0, random_state=10) # all data for training

	# train
	best = {}
	global ESTIMATORS
	estimators = ESTIMATORS.copy()
	if preferred_est is not None and preferred_est in ESTIMATORS:
		estimators = { preferred_est:ESTIMATORS[preferred_est] }
	else:
		if skip_est is not None:
			estimators = {k:v for k, v in estimators.iteritems() if k not in skip_est}
	print 'Trying estimators.. ', estimators.keys()
	for est in estimators:
		print 'Building {} ..'.format(est)
		steps = [('clf', estimators[est]['clf'])]
		if not abstract_data and scale:
			steps.insert(0, ('scaler', StandardScaler()))
		pipe = Pipeline(steps)
		params = {'clf__{}'.format(k):v for k, v in estimators[est]['param'].iteritems()}
		grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, refit=True, scoring=acc_scorer)
		grid_search.fit(X_train, y_train)
		best_est, best_param, best_score = grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
		best[est] = {'clf': best_est, 'best_score': best_score}
	return best

def build_model(X, y, outdir, estimator=None, suffix=None):
	"""
	Builds a number of models, unless estimator is specified indicating the 
	preferred estimator to fit, or specified estimators are requested to be skipped.

	* resulting model's pickled file is saved at outdir
	* suffix indicates the file suffix for the resulting models.
	"""
	best_models = find_best_model(X, y, preferred_est=estimator, \
					cv=CV, skip_est=['adaboost', 'svm', 'logistic'])

	# check performance
	print '\nBest model(s):'
	for i, best in enumerate(best_models):
		clf = best_models.get(best)
		print '{}. {}: {}'.format(i+1, best, clf)
		try:
			outfile = join(outdir, 'clf_{}_{}.pkl'.format(best, suffix))
			with open(outfile, 'wb') as g:
				pkl.dump(clf, g, protocol=pkl.HIGHEST_PROTOCOL)
			print 'Saved {} model: {}\n'.format(best, outfile)
		except IOError, e:
			raise e

def get_deduplicated_triples(triples_fname, target_dedup, person_dedup):
	"""Reads triples from the given file and deduplicates persons and 
	profession/nationality entities. Returns list of triples and ground truth vector."""
	tt = pd.read_table(triples_fname, sep=',', header=0)
	print 'Read input triples: {}'.format(tt.shape),
	tt = tt.dropna()
	print '.. After dropping NAs, #input triples: {}'.format(tt.shape)
	y = tt['human_relevance']
	tt = tt[['sub', 'obj']].to_dict(orient='list')
	triples = zip(tt['sub'], tt['obj'])

	# de-duplicate target: professions/nationalities
	target_all = pd.read_table(target_dedup, sep=',', header=0)
	target_all = target_all[['cup_node_name', 'node_name']]
	target_all.index = target_all['cup_node_name']
	target_all = target_all.drop('cup_node_name', axis=1).to_dict()['node_name']
	triples = [(t[0], target_all[t[1]]) if t[1] in target_all else t for t in triples]

	# de-duplicate persons
	persons_all = pd.read_table(person_dedup, sep=',', header=0)
	persons_all = persons_all[['cup_node_name', 'node_name']]
	persons_all.index = persons_all['cup_node_name']
	persons_all = persons_all.drop('cup_node_name', axis=1).to_dict()['node_name']
	triples = [(persons_all[t[0]], t[1]) if t[0] in persons_all else t for t in triples]
	print '=> Deduplicated triples: {}'.format(len(triples))
	return triples, y

if __name__ == '__main__':
	"""
	Example call:

	python model_building.py 
		-fex ../../data/processed/abstracts/fex_abs_profession_features_top20.pkl
		-train ../../data/processed/cup/profession_train.csv
		-target_dedup ../../data/processed/cup/professions_match.csv 
		-person_dedup ../../data/processed/cup/persons_match.csv 
		-est ordLogReg
		
	"""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-train', type=str, required=True, \
		dest='train', help='Training file.')
	parser.add_argument('-fex', type=str, dest='fex', required=True, \
		help='Previously saved AbstractFeatureExtractor object file.')
	parser.add_argument('-target_dedup', type=str, required=True, dest='target_dedup', 
		help='Path to match file, e.g. professions_match.csv')
	parser.add_argument('-person_dedup', type=str, required=True, dest='person_dedup',\
		help='Path to match file, e.g. persons_match.csv')
	parser.add_argument('-est', type=str, dest='estimator', default='randomforest', \
		help='Optional estimator to use. e.g. adaboost, ordLogReg, randomForest')
	parser.add_argument('-D', '--outdir', \
		help='Output directory where to save the best model.')
	args = parser.parse_args()
	print 

	args.train = abspath(expanduser(args.train))
	args.fex = abspath(expanduser(args.fex))
	args.person_dedup = abspath(expanduser(args.person_dedup))
	args.target_dedup = abspath(expanduser(args.target_dedup))
	for fname in [args.fex, args.train, args.target_dedup, args.person_dedup]:
		if not exists(fname) or not isfile(fname):
			raise Exception('Not a file or does not exist: %s' % fname)
	if args.outdir is None:
		args.outdir = dirname(args.fex)
	outdir = abspath(expanduser(args.outdir))
	if not exists(outdir) or not isdir(outdir):
		raise Exception('Not a directory or does not exist: %s' % outdir)
	args.outdir = outdir

	# Read input data
	triples, y = get_deduplicated_triples(args.train, args.target_dedup, args.person_dedup)
	
	# Load feature extractor object
	fex = AbstractFeatureExtractor(None, None, None)
	fex.load_feature_extractor(args.fex)
	X = fex.get_feature_matrix(triples)

	suffix = splitext(basename(args.fex))[0].split('fex_')[1]
	build_model(X, y, args.outdir, estimator=args.estimator, suffix=suffix)

	print '\nDone!\n'