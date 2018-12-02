# -*- coding: utf-8 -*-
""" Read in Gutenberg Poetry Corpus data and Kaggle Song Lyrics data, and write each line out
as phonetically transcribed by the CMU Pronouncing Dictionary"""
import ndjson
import cmudict
import re
import csv
from datetime import datetime

PUNCTUATION = [".",",","!","?",";"]

def load_and_transcribe_poetry(transcribed_dict, cmu):
	"""Load and transcribe the lines from the Gutenberg Poetry Corpus"""
	lines_dict = dict()

	with open('gutenberg_poetry.ndjson') as gberg_poetry:
		loaded_poetry = ndjson.load(gberg_poetry)

	for x in loaded_poetry:
		line = x['s']
		words = re.findall(r"[\w']+|[.,!?;]", line) #make punctuation separate words
		lines_dict[line] = words

	print("Poetry Loaded: {0}".format(datetime.now().time()))
	transcribe_lines(lines_dict, transcribed_dict, cmu)

def load_and_transcribe_lyrics(transcribed_dict, cmu):
	"""Load and transcribe the lyrics from the Kaggle lyrics dataset"""
	lines_dict = dict()
	with open('songdata.csv') as lyrics:
		rdr = csv.reader(lyrics)
		for row in rdr:
			if len(row) > 3: #in case any lyrics are blank
				song = row[3]
				lines = [s.strip() for s in song.splitlines()]
				lines = [s for s in lines if len(s) > 0] #remove blank newline lines
				for line in lines:
					words = re.findall(r"[\w']+|[.,!?;]", line) #make punctuation separate words
					lines_dict[line] = words
	print("Lyrics Loaded: {0}".format(datetime.now().time()))
	transcribe_lines(lines_dict, transcribed_dict, cmu)


def transcribe_lines(lines, transcribed_dict, cmu ):
	"""Take in a dictionary of line -> list of words in that line. Transcribe the words
	if possible, and add original lines and transcribed lines as K/V pairs in transcribed_dict"""
	for line, words in lines.items():
		transcription = []
		pronounceable = True
		for word in words:
			word = word.lower() #cmudict is all lowercased
			if word in PUNCTUATION:
				continue
			pron = cmu[word]
			if len(pron) == 0: #not in dict, throw out this line
				pronounceable = False
				break
			else:
				transcription.extend(pron[0]) #take the first pronunciation if there are multiple
		if pronounceable:
			#dict entries are orig line -> [transcribed phonemes], space-separated
			transcribed_dict[line] = ' '.join(transcription)


def main():
	print("Start time: {0}".format(datetime.now().time()))
	cmu = cmudict.dict()
	all_transcribed_lines = dict()

	load_and_transcribe_poetry(all_transcribed_lines, cmu)
	print("Poetry Transcribed: {0}".format(datetime.now().time()))
	load_and_transcribe_lyrics(all_transcribed_lines, cmu)
	print("Lyrics Transcribed: {0}".format(datetime.now().time()))

	with open('transcribed_data.csv', mode='w') as out_file:
		out_writer = csv.writer(out_file, delimiter=',', quotechar='"')
		for line, val in all_transcribed_lines.items():
			out_writer.writerow([line.encode('utf-8'), val])
	print("Written to file: {0}".format(datetime.now().time()))


if __name__ == "__main__":
	main()