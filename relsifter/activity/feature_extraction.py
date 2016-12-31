"""
Extracts features for a set of triples specified in an input file.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import shutil
import re

from time import time
from pandas import DataFrame, Series, merge
from datetime import date
from os.path import abspath, expanduser, dirname, isfile, isdir, exists,\
	join, basename, splitext
from collections import OrderedDict
from scipy.sparse import csr_matrix, vstack

from relsifter.datastructures.rgraph import Graph

class FeatureExtractor:
	"""FeatureExtractor for the WSDM Cup challenge."""
	def __init__(self, graph_path, shape, fmat_fname, rel='profession', top_k=5):
		"""
		graph_path: Path to the graph file(s).
		shape: (#nodes, #nodes, #relations) representing the graph.
		fmat_fname: Feature matrix file. If it exists, it will be read as-is,
					otherwise the parent directory needs to contain files 
					representing features for professions/nationalities
		"""
		self.relation = rel
		self.G = self.read_graph(graph_path, shape) # KG
		self.top_k = top_k # top k features
		# feature matrix
		if isdir(fmat_fname):
			fmat_fname = join(fmat_fname, '{}_feature_matrix_top{}.csv'.format(rel, top_k))
			# dictionary of prof/nationality -> file pairs 
			self.featureval_files = self.get_featureval_files(dirname(fmat_fname))
			top_features = self.read_features(self.featureval_files, self.top_k)
			
			self.feature_matrix = self.construct_feature_matrix(
				top_features, self.featureval_files, self.top_k
			)
			print '=> Feature matrix created: {}'.format(self.feature_matrix.shape)
			self.feature_matrix.to_csv(fmat_fname, sep=',', header=True, index=True)
			print '=> Feature matrix saved: {}'.format(fmat_fname)
		else:
			self.feature_matrix = pd.read_table(fmat_fname, sep=',', header=0, index_col=0)
			print '  * Feature matrix loaded: ({}) {}'.format(basename(fmat_fname), self.feature_matrix.shape)
		self.feature_idx = {int(rid):idx for idx, rid in enumerate(self.feature_matrix.columns)}
		self.feature_idx = OrderedDict(sorted(self.feature_idx.items(), key=lambda x: x[1]))

	def __str__(self):
		print "FeatureExtractor for relation '{}': {}".format(
			self.rel, self.feature_matrix.shape
		)

	def read_graph(self, graph_path, shape):
		"Reconstructs knowledge graph (KG) from files at the given directory."
		G = Graph.reconstruct(graph_path, shape=shape, sym=True, display=False)
		return G

	def read_features(self, featureval_files, top_k):
		"Constructs a set of top k features based on the feature files."
		features = dict()
		var = 'activity'
		try:
			for k, fname in featureval_files.iteritems():
				df = pd.read_table(fname, sep=',', header=0)
				df = df.iloc[:top_k]
				df.index =  df['activityId'].astype(np.int32)
				df = df.drop(df.columns[~df.columns.isin([var])], axis=1)
				top_k_feat = df.to_dict()[var] # relationId/activityId -> activity
				# add activity/relation as feature if not already in the set
				for rid, rname in top_k_feat.iteritems():
					if rid not in features:
						features[rid] = rname
			return features
		except IOError, e:
			raise e

	def construct_feature_matrix(self, features, featureval_files, top_k):
		"Constructs a feature data matrix for professions/nationalities."
		cols = sorted(features.keys())
		idx = sorted(featureval_files.keys())
		d = DataFrame(columns=cols) # empty data frame
		for k in idx:
			fname = featureval_files[k]
			df = pd.read_table(fname, sep=',', header=0)
			df.index =  df['activityId'].astype(np.int32)
			var = 'activityTFIDF'
			df['activityTFIDFnorm'] = df[var]/df[var].max()
			var = 'activityTFIDFnorm'
			df = df.drop(df.columns[~df.columns.isin([var])], axis=1)
			df = df.iloc[:top_k]
			df = df.to_dict()[var] # relationId/activityId -> value
			vec = np.array([df[r] if r in df else 0. for r in cols]).reshape((1, -1))
			d = d.append(DataFrame(vec, columns=cols))
		d.index = idx
		return d

	def get_featureval_files(self, fpath):
		"Creates a dictionary of prof/nat -> feature file paths."
		if not exists(fpath):
			raise Exception('Directory %s does not exist.' % fpath)
		if not isdir(fpath):
			raise Exception('%s is not a directory.' % fpath)
		featurevalfiles = dict()
		for f in os.listdir(fpath):
			if re.match('[0-9]+.csv', f) is not None:
				oid = int(f.split('.csv')[0])
				fname = join(fpath, f)
				if not exists(fname):
					raise Exception('File %s does not exist.' % fname)
				featurevalfiles[oid] = fname
		return featurevalfiles

	def features_for_triple(self, triple, as_dict=False, use_count=False):
		"""Extracts top k features (dict) for a given input triple (dict)."""
		# find the target 
		sid, oid = int(triple['sid']), int(triple['oid'])

		if self.feature_matrix is None:
			raise Exception('FeatureExtractor not initialized.')

		# read feature values for the given person
		rels, relcounts = np.unique(self.G.get_neighbors(sid)[0,:], return_counts=True)
		vec = np.zeros(self.feature_matrix.shape[1])
		for rid, cnt in zip(rels, relcounts):
			if rid in self.feature_idx:
				idx = self.feature_idx[rid]
				if use_count:
					vec[idx] = cnt
				else:
					vec[idx] = 1
		# print self.feature_matrix.columns[self.feature_matrix.loc[oid, :].values.nonzero()]
		fvec = np.multiply(vec, self.feature_matrix.loc[oid, :].values)
		if as_dict:
			fvec = dict(zip(self.feature_matrix.columns, fvec))
			fvec = dict(triple.items() + fvec.items())
		else:
			fvec = fvec.reshape((1,-1))
		return fvec

	def extract_features(self, triples_list, use_count=False):
		"Extracts features for a given set of triples, say for training set."
		df = DataFrame(columns=self.feature_idx.keys())
		fmat = []
		for triple in triples_list:
			fvec = self.features_for_triple(triple, use_count=use_count)
			fmat.append(csr_matrix(fvec))
		df = DataFrame(vstack(fmat).toarray(), columns=self.feature_idx.keys())
		return df

if __name__ == '__main__':
	"""
	# DBpedia
	python feature_extraction.py 
		-g ../../data/processed/kg/_undir
		-shape 6060993 6060993 663
		-feat ../../data/processed/cup/experiment1/professions
		-path ../../data/processed/cup/profession_train.csv
 	
 	# Wikidata
 	python feature_extraction.py  
 		-g ../../wikidata/processed/kg/_undir 
 		-shape 29413625 29413625 839 
 		-feat ../../wikidata/processed/cup/experiment4/professions 
 		-path ../../wikidata/processed/cup/profession_train.csv
	"""
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-g', type=str, required=True,
		dest='graphbase', help='Directory where graph vectors are located.')
	parser.add_argument('-shape', type=int, nargs='+', required=True, 
		dest='shape', help='Graph shape.')
	parser.add_argument('-feat', type=str, required=True, dest='feat', 
		help='Directory where professions/nationalities feature files are located.')
	parser.add_argument('-path', type=str, required=True,
		dest='path', help='Path to (training) triples file.')
	parser.add_argument('-outpath', type=str, 
			dest='outpath', help='Absolute path to the output directory.')
	parser.add_argument('-top', type=int, nargs='+', dest='topfeatures', \
			default=5, help='Top k features to use.')
	args = parser.parse_args()

	args.path = abspath(expanduser(args.path))
	args.graphbase = abspath(expanduser(args.graphbase))	
	args.feat = abspath(expanduser(args.feat))
	if args.outpath is not None:
		args.outpath = abspath(expanduser(args.outpath)) 
	else:
		args.outpath = dirname(args.feat)
	if not exists(args.path):
		raise Exception('File not found: %s' % args.path)
	if not isfile(args.path):
		raise ValueError('Input is not a file: %s' % args.path)
	for dname in [args.graphbase, args.feat, args.outpath]:
		if not exists(dname):
			raise Exception('Directory not found: %s' % dname)
		if not isdir(dname):
			raise Exception('Not a directory: %s' % dname)
	
	if isinstance(args.topfeatures, int):
		args.topfeatures = [args.topfeatures]
	rel = None
	if 'profession' in args.feat:
		rel = 'profession'
	elif 'nation' in args.feat:
		rel = 'nationality'
	else:
		raise Exception('Unrecognized relation type (profession/nationality).')

	# Read input data
	df = pd.read_table(args.path, sep=',', header=0)
	print 'Read: {}, {}'.format(df.shape, basename(args.path))
	df = df.dropna()
	print '=> After dropping records with NA: {}'.format(df.shape)
	
	# feature extraction for every value of top_k
	for top in args.topfeatures:
		# Feature extractor object
		fex = FeatureExtractor(args.graphbase, args.shape, args.feat, rel, top)

		# extract features
		t1 = time()
		df_features = fex.extract_features([r.to_dict() for _, r in df.iterrows()])
		df_features.index = df.index
		newdf = pd.concat([df, df_features], axis=1)
		print 'Time taken for feature extraction: {:.2f} secs.'.format(time() - t1)

		# save the extracted features
		outfile = join(args.outpath, '{}_features_top{}.csv'.format(
			splitext(basename(args.path))[0], top
		))
		newdf.to_csv(outfile, sep=',', header=True, index=False)
		print 'Saved features: {}\n'.format(outfile)

	print '\nDone!\n'
