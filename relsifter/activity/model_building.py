"""
Builds the best machine learning model for the input training data.
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
	}
}

# Scoring functions
def acc_fn(y, ypred, k=2, clip=True):
	ypred = np.asarray(ypred)
	y = np.asarray(y)
	if clip:
		ypred[np.in1d(ypred, [0, 1])] = 2
		ypred[np.in1d(ypred, [6, 7])] = 5
	return (np.abs(y - ypred) <= k).sum() / float(len(y))

def omitcols(d, first=7):
	return d.iloc[:, first:]

def find_best_model(X, y, cv=3, preferred_est=None, exclude_cols=7, clip=True):
	"""Tries different models on the given data and returns a list of best models (along with their best score) 
	for each parameter setting."""
	# scoring function
	scoring = make_scorer(acc_fn, greater_is_better=True, clip=clip)

	# make data 
	X_train, X_test, y_train, y_test = train_test_split(X, y, \
				test_size=0, random_state=10) # all data for training

	# train
	best = {}
	global ESTIMATORS
	estimators = ESTIMATORS.copy()
	if preferred_est is not None and preferred_est in ESTIMATORS:
		estimators = { preferred_est:ESTIMATORS[preferred_est] }
	print 'Trying estimators.. ', estimators.keys()
	for est in estimators:
		print 'Building {} ..'.format(est)
		# steps = [('scaler', StandardScaler()), ('clf', estimators[est]['clf'])]
		steps = [('clf', estimators[est]['clf'])]
		pipe = Pipeline(steps)
		params = {'clf__{}'.format(k):v for k, v in estimators[est]['param'].iteritems()}
		grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, refit=True, scoring=scoring)
		grid_search.fit(omitcols(X_train, first=exclude_cols), y_train)
		best_est, best_param, best_score = grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
		best[est] = {'clf': best_est, 'best_score': best_score}
	return best

def build_model(train, outdir, estimator=None, exclude_cols=7, suffix=None, save=True, **kwargs):
	"""
	Builds a number of models, unless estimator is specified indicating the 
	preferred estimator to fit.

	* resulting model's pickled file is saved at outdir
	* exclude_cols indicates the first exclude_cols columns to exclude.
	* suffix indicates the file suffix for the resulting models.
	"""
	X, y = train, train['human_relevance'].astype(np.int64)
	cv = kwargs['cv']
	best_models = find_best_model(
		X, y, cv=cv, preferred_est=estimator, exclude_cols=exclude_cols, clip=kwargs['clip']
	)
	sys.stdout.flush()

	# check performance
	print '\nBest model(s):'
	for i, best in enumerate(best_models):
		clf = best_models.get(best)
		if 'display' not in kwargs:
			print '{}. {}: {}'.format(i+1, best, clf)
		if save:
			try:
				outfile = join(outdir, 'clf_{}_{}.pkl'.format(best, suffix))
				with open(outfile, 'wb') as g:
					pkl.dump(clf, g, protocol=pkl.HIGHEST_PROTOCOL)
				print 'Saved {} model: {}\n'.format(best, outfile)
			except IOError, e:
				raise e
	return best_models

if __name__ == '__main__':
	"""
	Example call:
	
	# DBpedia
	python model_building.py 
		-train ../../data/processed/cup/experiment4/profession_train_features.csv
		-est randomForest

	# Wikidata
	python model_building.py 
		-train ../../wikidata/processed/cup/experiment4/profession_train_features_top5.csv
		-est randomForest -exclude 10

	# Wikidata: noclip
	python model_building.py  
		-train ../../wikidata/processed/cup/experiment4b/profession_train_features_top10.csv 
		-est randomForest -exclude 10 -noclip
		
	"""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-train', type=str, dest='train', help='Training file.')
	parser.add_argument('-est', type=str, dest='estimator', default='randomforest', \
		help='Optional estimator to use. e.g. adaboost, ordLogReg, randomForest')
	parser.add_argument('-exclude', type=int, dest='exclude_cols', default=7, \
		help='No. of initial columns to exclude.')
	parser.add_argument('-D', '--outdir', \
		help='Output directory where to save the best model.')
	parser.add_argument('-cv', '--cv', type=int, default=10, 
		help='No. of cross-validation folds.')
	parser.add_argument('-noclip', dest='noclip', action='store_true',
		help='Use clipping during model selection')
	args = parser.parse_args()
	print 

	train = abspath(expanduser(args.train))
	if not exists(train) or not isfile(train):
		raise Exception('Not a train file or does not exist: %s' % train)
	args.train = train
	if args.outdir is None:
		args.outdir = dirname(args.train)
	outdir = abspath(expanduser(args.outdir))
	if not exists(outdir) or not isdir(outdir):
		raise Exception('Not a directory or does not exist: %s' % outdir)
	args.outdir = outdir

	df = pd.read_table(args.train, sep=',', header=0)
	print 'Read training data: {} {}'.format(basename(args.train), df.shape)
	print 'Working on {} triples..'.format(df.shape)

	suffix = splitext(basename(args.train))[0]
	if args.noclip:
		suffix += '_noclip'
	build_model(
		df, args.outdir, estimator=args.estimator, cv=args.cv, \
		suffix=suffix, exclude_cols=args.exclude_cols, clip=(not args.noclip)
	)

	print '\nDone!\n'