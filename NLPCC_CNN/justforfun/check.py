#!/usr/bin/env python

import sys
import json

test_data = {}
def get_test_data(ifile):
	fi = open(ifile, 'r')
	for line in fi:
		js = json.loads(line.strip())
		query_id = js['query_id']
		test_data[query_id] = test_data.get(query_id, {})
		for passage in js['passages']:
			test_data[query_id][passage['passage_id']] = 1
	fi.close()

def check_submit(submit_file):
	cnt, error_lst = 0, []
	fi = open(submit_file, 'r')
	for line in fi:
		cnt += 1
		try:
			js = json.loads(line.strip())
		except:
			error_lst.append('illegal json at %d' %cnt)
			continue
		if not js.has_key('query_id') or not test_data.has_key(js['query_id']):
			error_lst.append('query_id invalid at %d' %cnt)
		else:
			if not js.has_key('ranklist') or not isinstance(js['ranklist'], list) or not js:
				error_lst.append('ranklist invalid at %d' %cnt)
			else:
				if len(js['ranklist']) != len(test_data[js['query_id']]):
					error_lst.append('length of ranklist not match at %d' %cnt)
				for itm in js['ranklist']:
					if not isinstance(itm, dict) or not itm:
						error_lst.append('ranklist itm invalid at %d' %cnt)
					else:
						if not itm.has_key('passage_id') or not test_data[js['query_id']].has_key(itm['passage_id']):
							error_lst.append('ranklist passage_id invalid at %d' %cnt)
						if not itm.has_key('rank') or not isinstance(itm['rank'], int):
							error_lst.append('ranklist passage_id invalid at %d' %cnt)
	fi.close()

	return error_lst

def do_work():
	test_file, submit_file = sys.argv[1:]
	get_test_data(test_file)
	errlog = check_submit(submit_file)
	if errlog:
		print 'ERROR: %s' %'\n'.join(errlog)
	else:
		print 'OK'

if __name__ == '__main__':
	do_work()
