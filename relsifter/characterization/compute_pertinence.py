"""
Computes the pertinence of relations for type-like relations.

This includes:
1) Pertinence by popularity
2) Pertinence by focus
3) Combined pertinence (produce of above two)
"""
import os
import sys
import numpy as np
import pandas as pd
import re

from argparse import ArgumentParser
from pandas import DataFrame, Series, merge
from os.path import join, exists, dirname, abspath, expanduser, \
	isfile, isdir, basename, splitext
from time import time
from datetime import date

from relsifter.datastructures.rgraph import Graph

relations = None
G = None
sid_people = None
pid_of_sid_people = None
rel_freq = None
cupbase = None

# ============ some helper functions  ============
def id_to_node(idx):
	"""Returns node name for a given node id."""
	tmp = nodes[nodes['node_id'] == idx]['node_name'].values
	if len(tmp) > 0:
		return tmp[0]
	
def node_to_id(node):
	"""Returns node id for a given node name."""
	tmp = nodes[nodes['node_name'] == node]['node_id'].values
	if len(tmp) > 0:
		return tmp[0]
	
def id_to_relation(idx):
	"""Returns relation name for a given relation id."""
	tmp = relations[relations['relation_id'] == idx]['relation'].values
	if len(tmp) > 0:
		return tmp[0]
	
def relation_to_id(rel):
	"""Returns relation id for a given relation name."""
	tmp = relations[relations['relation'] == rel]['relation_id'].values
	if len(tmp) > 0:
		return tmp[0]

def read_relations(relations_fname):
	"""
	Reads (relation id, relation name) pairs.
	"""
	assert exists(relations_fname)
	global relations
	relations = pd.read_table(relations_fname, sep=" ", header=0, usecols=[0,2], names=['relation_id', 'relation'])
	print 'Read successfully: {}, {}'.format(relations.shape, basename(relations_fname))

def read_graph(graphbase, shape):
	global G
	G = Graph.reconstruct(graphbase, shape=shape, sym=True, display=True)

def read_people(people_fname):
	global sid_people, pid_of_sid_people
	# Find all people in KG, set S
	# part 1: people having type dbo:Person in the KG
	type_id = relation_to_id('rdf:type')
	person_id = 21857894 # id of human/wd:Q5, check nodes.txt
	sid_people = G.get_neighbors(person_id, k=type_id)[1,:] # WORKS only for undirected graph. Logic changes for directed case.
	sid_people = set(np.unique(sid_people))
	print '#People in KG: {}'.format(len(sid_people))

	# part 2: people provided to us by the cup organizers
	cup_people = pd.read_table(people_fname, sep=',', header=0)
	cup_sid_people = set(cup_people['node_id'].values)
	print '#People provided by WSDM Cup org.: {}'.format(len(cup_sid_people))

	# merge the set
	sid_people = sid_people | cup_sid_people
	sid_people = {int(s) for s in sid_people}
	print '#People (KG + Cup): {}'.format(len(sid_people))

	# Find frequency of relations for all people in KG
	pid_of_sid_people = dict()
	for n in sid_people:
		relnbrs = G.get_neighbors(n)
		relids = np.unique(relnbrs[0,:])
		for rid in relids:
			pid_of_sid_people[rid] = pid_of_sid_people.get(rid, 0) + 1
	print '#Relations for all people in KG: {}'.format(len(pid_of_sid_people))

def generate_profession_features(method, infile):
	outpath = join(cupbase, method.__name__, 'professions')
	if not exists(outpath):
		os.makedirs(outpath)
		print '* Created: {}'.format(outpath)
	relsyn = []
	reltype = 'occupation'
	top = 5

	# read set of professions
	prof = pd.read_table(infile, sep=',', header=0)
	top_relations = {'R{}'.format(i+1):[] for i in xrange(top)}
	top_relations['profession_id'] = []
	top_relations['profession'] = []
	for idx, row in prof.iterrows():
		row = row.to_dict()
		outfile = join(outpath, '{}.csv'.format(int(row['node_id'])))
		df = method(reltype, oid=row['node_id'], relsynonyms=relsyn, display=False) # actual call to characterization
		top_relations['profession'].append(row['node_name'])
		top_relations['profession_id'].append(row['node_id'])
		for i in xrange(top):
			if df.shape[0] > 0:
				rel, relid = df.iloc[i,:]['activity'], df.iloc[i,:]['activityId']
			else:
				rel, relid = None, None
			top_relations['R{}'.format(i+1)].append(rel)
		df.to_csv(outfile, sep=',', header=True, index=False)
		print '{}. Saved: {}'.format(idx+1, outfile)

	# save top relations
	top_relations = DataFrame.from_dict(top_relations, orient='columns')
	cols = ['profession_id', 'profession'] + ['R{}'.format(i+1) for i in xrange(top)]
	top_relations = top_relations[cols]
	top_relations_file = join(outpath, 'top_{}_relations_for_professions.csv'.format(top))
	top_relations.to_csv(top_relations_file, sep=',', header=True, index=False)
	print '\nSaved: top-{} relations for each profession: {}'.format(top, top_relations_file)

def generate_nationality_features(method, infile):
	outpath = join(cupbase, method.__name__, 'nationalities')
	if not exists(outpath):
		os.makedirs(outpath)
		print '* Created: {}'.format(outpath)
	relsyn = []
	reltype = 'country of citizenship'
	top = 5

	# read set of nationalities
	prof = pd.read_table(infile, sep=',', header=0)
	top_relations = {'R{}'.format(i+1):[] for i in xrange(top)}
	top_relations['nationality_id'] = []
	top_relations['nationality'] = []
	for idx, row in prof.iterrows():
		row = row.to_dict()
		outfile = join(outpath, '{}.csv'.format(int(row['node_id'])))
		df = method(reltype, oid=row['node_id'], relsynonyms=relsyn, display=False)
		top_relations['nationality'].append(row['node_name'])
		top_relations['nationality_id'].append(row['node_id'])
		for i in xrange(top):
			if df.shape[0] > 0:
				rel, relid = df.iloc[i,:]['activity'], df.iloc[i,:]['activityId']
			else:
				rel, relid = None, None
			top_relations['R{}'.format(i+1)].append(rel)
		df.to_csv(outfile, sep=',', header=True, index=False)
		print '{}. Saved: {}'.format(idx+1, outfile)

	# save top relations
	top_relations = DataFrame.from_dict(top_relations, orient='columns')
	cols = ['nationality_id', 'nationality'] + ['R{}'.format(i+1) for i in xrange(top)]
	top_relations = top_relations[cols]
	top_relations_file = join(outpath, 'top_{}_relations_for_nationalities.csv'.format(top))
	top_relations.to_csv(top_relations_file, sep=',', header=True, index=False)
	print '\nSaved: top-{} relations for each nationality: {}'.format(top, top_relations_file)

def popular_pertinence(reltype, oid, relsynonyms=None, display=True):
	global sid_people, pid_of_sid_people
	reltypes = [reltype] + list(relsynonyms)

	# Find relation ids for this relation type and its synonyms
	pids = [relation_to_id(k) for k in reltypes]

	# Find people having these relation types, set A
	sid_with_reltype = set()
	for pid in pids:
		relnbrs = G.get_neighbors(oid, pid) # Note: works only for undirected graph. logic will change for directed graph.
		relnbrs = set(relnbrs[1,:]) & set(sid_people) # logical 'and' to ensure nbrs of target are people.
		sid_with_reltype |= relnbrs
	if display:
		print '#People having this relation, set A: {}'.format(len(sid_with_reltype))

	# Find all relations of people in A, set R, and 
	# TF - #People in A having relation r, dict: r -> #people with r in A
	pid_of_sid_with_reltype = dict() #{key:0 for key in relations['relation_id'].values}
	for n in sid_with_reltype:
		relnbrs = G.get_neighbors(n)
		relids = np.unique(relnbrs[0,:])
		for rid in relids:
			pid_of_sid_with_reltype[rid] = pid_of_sid_with_reltype.get(rid, 0) + 1
	rels = list(set(pid_of_sid_with_reltype.keys()) - set(reltypes)) # set R
	if display:
		print '#Relations of people in A: {}'.format(len(rels))

	# IDf - dict: r -> number of people in S / number of people in S with relation r
	N = len(sid_people)
	idf_local = {r: np.log(N/float(pid_of_sid_people[r])) for r in rels}

	df = DataFrame.from_dict({
			'activityId': rels, 
			'activity': [id_to_relation(r) for r in rels],
			'activityTF': [np.log(1 + pid_of_sid_with_reltype[r]) for r in rels],
			'activityIDF': [idf_local[r] for r in rels]
	})
	df['activityTFIDF'] = df['activityTF'] * df['activityIDF']
	df = df.sort_values(by='activityTFIDF', ascending=False)
	df = df[['activityId', 'activity', 'activityTF', 'activityIDF', 'activityTFIDF']]
	return df

def read_relational_frequency():
	global rel_freq, sid_people
	if rel_freq is not None:
		return rel_freq
	# frequency of relation across all people
	rel_freq = dict()
	for n in sid_people:
	    relnbrs = G.get_neighbors(n)
	    relids, relcounts = np.unique(relnbrs[0,:], return_counts=True)
	    for rid, rcnt in zip(relids, relcounts):
	        rel_freq[rid] = rel_freq.get(rid, 0) + rcnt

def relational_pertinence(reltype, oid, relsynonyms=None, display=False):
	global sid_people, pid_of_sid_people
	reltypes = [reltype] + list(relsynonyms)
	# Find relation ids for this relation type and its synonyms
	pids = [relation_to_id(k) for k in reltypes]

	# Cache relational frequency
	read_relational_frequency()

	# Find people having these relation types
	sid_with_reltype = set()
	for pid in pids:
		relnbrs = G.get_neighbors(oid, pid) # Note: works only for undirected graph. logic will change for directed graph.
		relnbrs = set(relnbrs[1,:]) & set(sid_people)
		sid_with_reltype |= relnbrs

	# IDF of people with this reltype
	N = len(sid_with_reltype)
	tmp = dict()
	for n in sid_with_reltype:
		rels = np.unique(G.get_neighbors(n)[0,:])
		for r in rels:
			tmp[r] = tmp.get(r, 0) + 1
	idf_local_reltype = {k: np.log(N/float(cnt)) for k, cnt in tmp.iteritems()}

	# Identify other relations (with their frequency) that these people are involved in
	pid_of_sid_with_reltype = dict() #{key:0 for key in relations['relation_id'].values}
	for n in sid_with_reltype:
		relnbrs = G.get_neighbors(n)
		relids, relcounts = np.unique(relnbrs[0,:], return_counts=True)
		for rid, rcnt in zip(relids, relcounts):
			pid_of_sid_with_reltype[rid] = pid_of_sid_with_reltype.get(rid, 0) + rcnt

	N = len(sid_people)
	idf_local = {r: np.log(N/float(pid_of_sid_people[r])) for r in rels}
	
	# Running set of activities for this relationtype(s)
	df = DataFrame.from_dict({
			'activityId': pid_of_sid_with_reltype.keys(), 
			'activity': [id_to_relation(r) for r in pid_of_sid_with_reltype.keys()],
			'activityTF': [cnt for cnt in pid_of_sid_with_reltype.values()],
	})
	df['activityTF'] = df['activityTF'] / df['activityTF'].sum()
	df['activityTF_avg'] = [df.loc[df['activityId']==rid, 'activityTF'].values[0]/float(len(sid_with_reltype)) for rid in pid_of_sid_with_reltype]
	outofset_freq = {k:(v+1) if k not in pid_of_sid_with_reltype else 1+v-pid_of_sid_with_reltype[k] for k, v in rel_freq.iteritems()}
	tot = np.sum(outofset_freq.values())
	# outofset_freq = {k:v/float(tot) for k, v in outofset_freq.iteritems()}
	idf = {k:np.log(tot/float(v)) for k, v in outofset_freq.iteritems()}
	tf = {k:np.log(1 + pid_of_sid_with_reltype[k]) for k in pid_of_sid_with_reltype}
	# df['activityIDF'] = [(pid_of_sid_with_reltype[r])/(float(outofset_freq[r])) for r in pid_of_sid_with_reltype]
	# df = df[['activityId', 'activity', 'activityTF', 'activityTF_avg', 'activityIDF']]
	# df['activityTFIDF'] = df['activityIDF'] 
	df['activityTFIDF'] = [tf[r] * idf[r] for r in pid_of_sid_with_reltype]
	df = df.sort_values(by='activityTFIDF', ascending=False)
	return df

def combined_pertinence(reltype, oid, relsynonyms=None, display=False):
	"Combines experiment1 and experiment2 approach. Ignores most input parameters except the target."
	global cupbase
	pr_professions = join(cupbase, 'popular_pertinence/professions/')
	rr_professions = join(cupbase, 'relational_pertinence/professions/')
	pr_nationalities = join(cupbase, 'popular_pertinence/nationalities/')
	rr_nationalities = join(cupbase, 'relational_pertinence/nationalities/')
	
	if exists(pr_professions):
		fnames_professions = {int(f.split('.csv')[0]):f for f in os.listdir(pr_professions) if re.match('[0-9]+.csv', f) is not None}
	if exists(pr_nationalities):
		fnames_nationalities = {int(f.split('.csv')[0]):f for f in os.listdir(pr_nationalities) if re.match('[0-9]+.csv', f) is not None}
	reltypes = [reltype] + list(relsynonyms)

	# Find relation ids for this relation type and its synonyms
	pids = [relation_to_id(k) for k in reltypes]

	if 'occupation' in reltypes:
		exp1 = pr_professions
		exp2 = rr_professions
		fnames = fnames_professions
	if 'country of citizenship' in reltypes:
		exp1 = pr_nationalities
		exp2 = rr_nationalities
		fnames = fnames_nationalities
	df1 = pd.read_table(join(exp1, fnames[oid]), sep=',', header=0)
	df2 = pd.read_table(join(exp2, fnames[oid]), sep=',', header=0)
	d = pd.merge(df1, df2, left_on=['activityId', 'activity'], right_on=['activityId', 'activity'], how="inner", suffixes=('_1', "_2"))
	assert df1.shape[0] == df2.shape[0] and df1.shape[0] == d.shape[0]
	d['activityTFIDF'] = d['activityTFIDF_1'] * d['activityTFIDF_2']
	d = d.sort_values(by='activityTFIDF', ascending=False)
	return d

def compute_pertinence(graphbase, shape, relations_fname, people_fname, infile, output, pertinence):
	global cupbase
	cupbase = output

	# Depending on pertinence
	if pertinence == 'popular':
		method = popular_pertinence
	elif pertinence == 'relational':
		method = relational_pertinence
	elif pertinence == 'combined':
		method = combined_pertinence

	# Read relations
	read_relations(relations_fname)
	sys.stdout.flush()

	if pertinence in ('popular', 'relational'):
		# Read graph
		read_graph(graphbase, shape)
		sys.stdout.flush()

		# Read people
		read_people(people_fname)	

	# Generate profession / nationality
	if 'profession' in infile:
		generate_profession_features(method, infile)
	elif 'nation' in infile:
		generate_nationality_features(method, infile)

if __name__ == '__main__':
	"""
	Example call:

	## Wikidata
	python compute_pertinence.py
		-kg ../model/kg/_undir/
		-rel ../model/kg/relations.txt
		-shape 29413625 29413625 839
		-R 'popular'
		-ppl ../../wikidata/processed/cup/persons_match.csv
		-i ../../wikidata/processed/cup/professions_match.csv
		-o ../../output/
	"""
	parser = ArgumentParser(description=__doc__)
	parser.add_argument('-kg', '--kg', required=True, \
		help='Absolute path to the knowledge graph directory.')
	parser.add_argument('-rel', '--rel', required=True, \
		help='Absolute path to the relations file.')
	parser.add_argument('-shape', '--shape', nargs='+', type=int, required=True, \
		help='Shape representing the knowledge graph.')
	parser.add_argument('-R', '--pertinence', required=True, \
		help='Relevance type: one of popular, relational, combined')
	parser.add_argument('-ppl', '--people', required=True, \
		help='Input file containing the persons.')
	parser.add_argument('-i', '--input', required=True, \
		help='Input file containing the profession/nationality triples.')
	parser.add_argument('-o', '--output', required=True, \
		help='Output directory. Will be created if non-existent.')
	args = parser.parse_args()
	print ''

	args.kg = abspath(expanduser(args.kg))
	args.rel = abspath(expanduser(args.rel))
	args.shape = tuple(args.shape)
	args.people = abspath(expanduser(args.people))
	args.input = abspath(expanduser(args.input))
	args.output = abspath(expanduser(args.output))
	if args.pertinence not in ('popular', 'relational', 'combined'):
		raise Exception('Unrecognized pertinence type: %s' % args.pertinence)
	
	compute_pertinence(
		args.kg, args.shape, args.rel, args.people, args.input, 
		args.output, args.pertinence
	)

	print '\nDone!\n'