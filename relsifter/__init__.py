import sys
import os

from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
	
_ROOT = abspath(dirname(__file__))
LOG = abspath(expanduser('~/logs/'))

if not exists(LOG):
	os.mkdir(LOG)
	print '** Created log directory: {}'.format(LOG)

class Logger(object):
	def __init__(self, filename="default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def __getattr__(self, attr):
		return getattr(self.terminal, attr)

def get_path(fname):
	return join(_ROOT, 'model', fname)

GRAPH_PATH = get_path('kg/_undir')
SHAPE = (29413625, 29413625, 839)
CLIP = False

if CLIP:
	#============== CLIPPED ==============
	DATAMAP = {
		'profession': {
			'rel': 'profession',
			'kb': get_path('profession/profession_kb_match.pkl'),
			'kg_features': get_path('profession/profession_feature_matrix_top100.csv'),
			'clf': get_path('profession/clf_randomForest_profession_train_features_top100.pkl'),
			'top_k': 100
		},
		'nationality': {
			'rel': 'nationality',
			'kb': get_path('nationality/nationality_kb_match.pkl'),
			'kg_features': get_path('nationality/nationality_feature_matrix_top5.csv'),
			'clf': get_path('nationality/clf_ordLogReg_nationality_train_features_top5.pkl'),
			'top_k': 5
		}
	}
else:
	#============== UNCLIPPED ==============
	DATAMAP = {
		'profession': {
			'rel': 'profession',
			'kb': get_path('profession/profession_kb_match.pkl'),
			'kg_features': get_path('profession/profession_feature_matrix_top100.csv'),
			'clf': get_path('profession/clf_randomForest_profession_train_features_top100_noclip.pkl'),
			'top_k': 100
		},
		'nationality': {
			'rel': 'nationality',
			'kb': get_path('nationality/nationality_kb_match.pkl'),
			'kg_features': get_path('nationality/nationality_feature_matrix_top50.csv'),
			'clf': get_path('nationality/clf_ordLogReg_nationality_train_features_top50_noclip.pkl'),
			'top_k': 50
		}
	}
