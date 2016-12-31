import sys
import os
import itertools
import subprocess

from os.path import expanduser, abspath, exists, join, basename

def check_identical_columns(filename1, filename2):
	"Assertions to ensure columns 1 and 2 are identical for input and output files."
	i = 0
	with open(filename1) as file1, open(filename2) as file2:
		for line1, line2 in itertools.izip_longest(file1, file2):
			i+= 1
			try:
				assert line1 != None, "#lines(file1) < #lines(file2)"
				assert line2 != None, "#lines(file2) < #lines(file1)"
				cols1, cols2 = line1.split("\t"), line2.split("\t")
				line_str = ", at line " + str(i)
				assert cols1[0] == cols2[0], "col1(file1) != col1(file2)" + line_str
			except Exception, e:
				print line1, line2
				raise e

def _execute(infile):
	"Executes the relsifter on the given input file."
	if "samphire" in expanduser('~'):
		cmd = abspath(expanduser('~/anaconda2/bin/relsifter'))
	else:
		cmd = abspath(expanduser('~/anaconda/bin/relsifter'))
	assert exists(cmd)
	outdir = abspath(expanduser('../../results/'))
	outfile = join(outdir, basename(infile))
	script = "{} -i {} -o {}".format(cmd, infile, outdir)
	print 'Launching..\n'
	print script
	retcode = subprocess.call(script, shell=True)
	check_identical_columns(infile, outfile)
	print ''

def test_profession_train():
	"Test for profession train set."
	infile = abspath(expanduser('../../wikidata/raw/cup/profession.train'))
	assert exists(infile)
	print '=> Testing {}'.format(infile)
	_execute(infile)

def test_nationality_train():
	"Test for nationality train set."
	infile = abspath(expanduser('../../wikidata/raw/cup/nationality.train'))
	assert exists(infile)
	print '=> Testing {}'.format(infile)
	_execute(infile)

if __name__ == '__main__':
	test_profession_train()
	test_nationality_train()
