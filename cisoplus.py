#!/usr/bin/env python2.7

from __future__ import print_function
import os
import struct
import sys
import zlib
import urllib
import pprint
import array
from itertools import chain
import os.path

try:
	from cStringIO import StringIO
except ImportError:
	from StringIO import StringIO

CISO_MAGIC = 0x4F534943 # CISO
CISO_HEADER_SIZE = 0x18 # 24
CISO_BLOCK_SIZE = 0x800 # 2048
CISO_HEADER_FMT = '<LLQLBBxx' # Little endian
CISO_WBITS = -15 # Maximum window size, suppress gzip header check.
CISO_PLAIN_BLOCK = 0x80000000
COM_THRESHOLD = 0.7
SECTOR_SIZE = 2048

#assert(struct.calcsize(CISO_HEADER_FMT) == CISO_HEADER_SIZE)

class ISO9660IOError(IOError):


	def __init__(self, path):
		self.path = path

	def __str__(self):
		return "Path not found: %s" % self.path

class ISO9660(object):

	PMF_HEADER = 'PSMF'
	AT3_HEADER = 'RIFF'
	PNG_HEADER = '\x89PNG'


	def __init__(self, url):
		self._buff  = None #input buffer
		self._root  = None #root node
		self._pvd   = {}   #primary volume descriptor
		self._paths = []   #path table
		self._media_file = []
		self._media_pos = []
		self._media_block = []
		self._vaccum_file = [
			'/PSP_GAME/SYSDIR/UPDATE/DATA.BIN', 
			'/PSP_GAME/SYSDIR/UPDATE/EBOOT.BIN']
		self._vaccum_block = []

		self._url   = url
		#if self._get_sector is None: #it might have been set by a subclass
		if '_get_sector' not in dir(self) or self._get_sector is None: #it might have been set by a subclass
			self._get_sector = self._get_sector_url if url.startswith('http') else self._get_sector_file

		### Volume Descriptors
		sector = 0x10
		while True:
			self._get_sector(sector, SECTOR_SIZE)
			sector += 1
			ty = self._unpack('B')

			if ty == 1:
				self._unpack_pvd()
			elif ty == 255:
				break
			else:
				continue

		### Path table
		l0 = self._pvd['path_table_size']
		self._get_sector(self._pvd['path_table_l_loc'], l0)

		while l0 > 0:
			p = {}
			l1 = self._unpack('B')
			l2 = self._unpack('B')
			p['ex_loc'] = self._unpack('<I')
			p['parent'] = self._unpack('<H')
			p['name']   = self._unpack_string(l1)
			if p['name'] == '\x00':
				p['name'] = ''

			if l1%2 == 1:
				self._unpack('B')

			self._paths.append(p)

			l0 -= 8 + l1 + (l1 % 2)

		assert l0 == 0

	##
	## Generator listing available files/folders
	##

	def tree(self, get_files = True):
		if get_files:
			gen = self._tree_node(self._root)
		else:
			gen = self._tree_path('', 1)

		#yield '/'
		for i in gen:
			yield i

	def _tree_path(self, name, index):
		spacer = lambda s: "%s/%s" % (name, s)
		for i, c in enumerate(self._paths):
			if c['parent'] == index and i != 0:
				yield spacer(c['name'])
				for d in self._tree_path(spacer(c['name']), i+1):
					yield d

	def _tree_node(self, node):
		spacer = lambda s: "%s/%s" % (node['name'], s)
		for c in list(self._unpack_dir_children(node)):
			yield spacer(c['name'])
			if c['flags'] & 2:
				for d in self._tree_node(c):
					yield spacer(d)

	##
	## Retrieve file contents as a string
	##
	def get_file_pos(self, path):
		#path = path.upper().strip('/').split('/')
		path = path.strip('/').split('/')
		path, filename = path[:-1], path[-1]

		if len(path)==0:
			parent_dir = self._root
		else:
			try:
				parent_dir = self._dir_record_by_table(path)
			except ISO9660IOError:
				try:
					parent_dir = self._dir_record_by_root(path)
				except:
					return None

		f = self._search_dir_children(parent_dir, filename)

		return (f['ex_loc'], f['ex_len'])
	
	def scan_media_file(self):
		header_size = 4
		with open(self._url, 'rb') as fh:
			for f in self.tree():
				s_l = self.get_file_pos(f)
				if s_l is not None:
					start_sec,  length = s_l
				if length < SECTOR_SIZE:
					continue
				if f.upper() in self._vaccum_file:
					self._vaccum_block.append((start_sec, start_sec+length/SECTOR_SIZE-1))
					continue
				start = start_sec * SECTOR_SIZE
				fh.seek(start)
				self._buff = StringIO(fh.read(length))
				if self._unpack_raw(header_size) in \
						(self.PMF_HEADER, self.AT3_HEADER, self.PNG_HEADER):
					self._media_file.append(f)
					self._media_block.append((start_sec, start_sec+length/SECTOR_SIZE+1))
					self._media_pos.append((start, start+length))

	def get_file(self, path):

		self._get_sector(*self.get_file_pos(path))
		return self._unpack_raw(f['ex_len'])

	##
	## Methods for retrieving partial contents
	##

	def _get_sector_url(self, sector, length):
		start = sector * SECTOR_SIZE
		if self._buff:
			self._buff.close()
		opener = urllib.FancyURLopener()
		opener.http_error_206 = lambda *a, **k: None
		opener.addheader("Range", "bytes=%d-%d" % (start, start+length-1))
		self._buff = opener.open(self._url)

	def _get_sector_file(self, sector, length):
		with open(self._url, 'rb') as f:
			f.seek(sector*SECTOR_SIZE)
			self._buff = StringIO(f.read(length))

	##
	## Return the record for final directory in a path
	##

	def _dir_record_by_table(self, path):
		for e in self._paths[::-1]:
			search = list(path)
			f = e
			while f['name'] == search[-1]:
				search.pop()
				f = self._paths[f['parent']-1]
				if f['parent'] == 1:
					e['ex_len'] = SECTOR_SIZE #TODO
					return e

		raise ISO9660IOError(path)

	def _dir_record_by_root(self, path):
		current = self._root
		remaining = list(path)

		while remaining:
			current = self._search_dir_children(current, remaining[0])

			remaining.pop(0)

		return current

	##
	## Unpack the Primary Volume Descriptor
	##

	def _unpack_pvd(self):
		self._pvd['type_code']                     = self._unpack_string(5)
		self._pvd['standard_identifier']           = self._unpack('B')
		self._unpack_raw(1)                        #discard 1 byte
		self._pvd['system_identifier']             = self._unpack_string(32)
		self._pvd['volume_identifier']             = self._unpack_string(32)
		self._unpack_raw(8)                        #discard 8 bytes
		self._pvd['volume_space_size']             = self._unpack_both('i')
		self._unpack_raw(32)                       #discard 32 bytes
		self._pvd['volume_set_size']               = self._unpack_both('h')
		self._pvd['volume_seq_num']                = self._unpack_both('h')
		self._pvd['logical_block_size']            = self._unpack_both('h')
		self._pvd['path_table_size']               = self._unpack_both('i')
		self._pvd['path_table_l_loc']              = self._unpack('<i')
		self._pvd['path_table_opt_l_loc']          = self._unpack('<i')
		self._pvd['path_table_m_loc']              = self._unpack('>i')
		self._pvd['path_table_opt_m_loc']          = self._unpack('>i')
		_, self._root = self._unpack_record()      #root directory record
		self._pvd['volume_set_identifer']          = self._unpack_string(128)
		self._pvd['publisher_identifier']          = self._unpack_string(128)
		self._pvd['data_preparer_identifier']      = self._unpack_string(128)
		self._pvd['application_identifier']        = self._unpack_string(128)
		self._pvd['copyright_file_identifier']     = self._unpack_string(38)
		self._pvd['abstract_file_identifier']      = self._unpack_string(36)
		self._pvd['bibliographic_file_identifier'] = self._unpack_string(37)
		self._pvd['volume_datetime_created']       = self._unpack_vd_datetime()
		self._pvd['volume_datetime_modified']      = self._unpack_vd_datetime()
		self._pvd['volume_datetime_expires']       = self._unpack_vd_datetime()
		self._pvd['volume_datetime_effective']     = self._unpack_vd_datetime()
		self._pvd['file_structure_version']        = self._unpack('B')

	##
	## Unpack a directory record (a listing of a file or folder)
	##

	def _unpack_record(self, read=0):
		l0 = self._unpack('B')

		if l0 == 0:
			return read+1, None

		l1 = self._unpack('B')

		d = dict()
		d['ex_loc']               = self._unpack_both('I')
		d['ex_len']               = self._unpack_both('I')
		d['datetime']             = self._unpack_dir_datetime()
		d['flags']                = self._unpack('B')
		d['interleave_unit_size'] = self._unpack('B')
		d['interleave_gap_size']  = self._unpack('B')
		d['volume_sequence']      = self._unpack_both('h')

		l2 = self._unpack('B')
		d['name'] = self._unpack_string(l2).split(';')[0]
		if d['name'] == '\x00':
			d['name'] = ''

		if l2 % 2 == 0:
			self._unpack('B')

		t = 34 + l2 - (l2 % 2)

		e = l0-t
		if e>0:
			extra = self._unpack_raw(e)

		return read+l0, d

	#Assuming d is a directory record, this generator yields its children
	def _unpack_dir_children(self, d):
		sector = d['ex_loc']
		read = 0
		self._get_sector(sector, 2048)

		read, r_self = self._unpack_record(read)
		read, r_parent = self._unpack_record(read)

		while read < r_self['ex_len']: #Iterate over files in the directory
			if read % 2048 == 0:
				sector += 1
				self._get_sector(sector, 2048)
			read, data = self._unpack_record(read)

			if data == None: #end of directory listing
				to_read = 2048 - (read % 2048)
				self._unpack_raw(to_read)
				read += to_read
			else:
				yield data

	#Search for one child amongst the children
	def _search_dir_children(self, d, term):
		for e in self._unpack_dir_children(d):
			if e['name'] == term:
				return e

		raise ISO9660IOError(term)
	##
	## Datatypes
	##

	def _unpack_raw(self, l):
		return self._buff.read(l)

	#both-endian
	def _unpack_both(self, st):
		a = self._unpack('<'+st)
		b = self._unpack('>'+st)
		assert a == b
		return a

	def _unpack_string(self, l):
		return self._buff.read(l).rstrip(' ')

	def _unpack(self, st):
		if st[0] not in ('<','>'):
			st = '<' + st
		d = struct.unpack(st, self._buff.read(struct.calcsize(st)))
		if len(st) == 2:
			return d[0]
		else:
			return d

	def _unpack_vd_datetime(self):
		return self._unpack_raw(17) #TODO

	def _unpack_dir_datetime(self):
		return self._unpack_raw(7) #TODO

def get_terminal_size(fd=sys.stdout.fileno()):
	try:
		import fcntl, termios
		hw = struct.unpack("hh", fcntl.ioctl(
			fd, termios.TIOCGWINSZ, '1234'))
	except:
		try:
			hw = (os.environ['LINES'], os.environ['COLUMNS'])
		except:
			hw = (25, 80)
	return hw

(console_height, console_width) = get_terminal_size()

def seek_and_read(f, pos, size):
	f.seek(pos, os.SEEK_SET)
	return f.read(size)

def parse_header_info(header_data):
	(magic, header_size, total_bytes, block_size,
			ver, align) = header_data
	if magic == CISO_MAGIC:
		ciso = {
			'magic': magic,
			'magic_str': ''.join(
				[chr(magic >> i & 0xFF) for i in (0,8,16,24)]),
			'header_size': header_size,
			'total_bytes': total_bytes,
			'block_size': block_size,
			'ver': ver,
			'align': align,
			'total_blocks': int(total_bytes / block_size),
			}
		ciso['index_size'] = (ciso['total_blocks'] + 1) * 4
	else:
		raise Exception("Not a CISO file.")
	return ciso

def update_progress(progress):
	barLength = console_width - len("Progress: 100% []") - 1
	block = int(round(barLength*progress)) + 1
	text = "\rProgress: [{blocks}] {percent:.0f}%".format(
			blocks="#" * block + "-" * (barLength - block),
			percent=progress * 100)
	sys.stdout.write(text)
	sys.stdout.flush()

def decompress_cso(infile, outfile):
	with open(outfile, 'wb') as fout:
		with open(infile, 'rb') as fin:
			data = seek_and_read(fin, 0, CISO_HEADER_SIZE)
			header_data = struct.unpack(CISO_HEADER_FMT, data)
			ciso = parse_header_info(header_data)

			# Print some info before we start
			print("Decompressing '{}' to '{}'".format(
				infile, outfile))
			for k, v in ciso.items():
				print("{}: {}".format(k, v))

			# Get the block index
			block_index = [struct.unpack("<I", fin.read(4))[0]
					for i in
					range(0, ciso['total_blocks'] + 1)]

			percent_period = ciso['total_blocks'] / 100
			percent_cnt = 0

			for block in range(0, ciso['total_blocks']):
				#print("block={}".format(block))
				index = block_index[block]
				plain = index & 0x80000000
				index &= 0x7FFFFFFF
				read_pos = index << (ciso['align'])
				#print("index={}, plain={}, read_pos={}".format(
				#	index, plain, read_pos))

				if plain:
					read_size = ciso['block_size']
				else:
					index2 = block_index[block + 1] & 0x7FFFFFFF
					read_size = (index2 - index) << (ciso['align'])

				raw_data = seek_and_read(fin, read_pos, read_size)
				raw_data_size = len(raw_data)
				if raw_data_size != read_size:
					#print("read_size={}".format(read_size))
					#print("block={}: read error".format(block))
					sys.exit(1)

				if plain:
					decompressed_data = raw_data
				else:
					decompressed_data = zlib.decompress(raw_data, CISO_WBITS)

				# Write decompressed data to outfile
				fout.write(decompressed_data)

				# Progress bar
				percent = int(round((block / (ciso['total_blocks'] + 1)) * 100))
				if percent > percent_cnt:
					update_progress((block / (ciso['total_blocks'] + 1)))
					percent_cnt = percent
		# close infile
	# close outfile
	return True

def check_file_size(f):
	f.seek(0, os.SEEK_END)
	file_size = f.tell()
	ciso = {
			'magic': CISO_MAGIC,
			'ver': 1,
			'block_size': CISO_BLOCK_SIZE,
			'total_bytes': file_size,
			'total_blocks': int(file_size / CISO_BLOCK_SIZE),
			'align': 0,
			}
	f.seek(0, os.SEEK_SET)
	return ciso

def write_cso_header(f, ciso):
	f.write(struct.pack(CISO_HEADER_FMT,
		ciso['magic'],
		CISO_HEADER_SIZE,
		ciso['total_bytes'],
		ciso['block_size'],
		ciso['ver'],
		ciso['align']
		))

def write_block_index(f, block_index):
	for index, block in enumerate(block_index):
		try:
			f.write(struct.pack('<I', block))
		except Exception as e:
			print("Writing block={} with data={} failed.".format(
				index, block))
			print(e)
			sys.exit(1)

def compress_iso(infile, outfile, compression_level, nocom_ranges=[], vaccum_ranges=[]):
	with open(outfile, 'wb') as fout:
		with open(infile, 'rb') as fin:
			print("Compressing '{}' to '{}'".format(
				infile, outfile))

			ciso = check_file_size(fin)
			for k, v in ciso.items():
				print("{}: {}".format(k, v))
			print("compress level: {}".format(compression_level))

			write_cso_header(fout, ciso)
			block_index = [0x00] * (ciso['total_blocks'] + 1)

			# Write the dummy block index for now.
			write_block_index(fout, block_index)

			write_pos = fout.tell()
			align_b = 1 << ciso['align']
			align_m = align_b - 1

			# Alignment buffer is unsigned char.
			alignment_buffer = struct.pack('<B', 0x00) * 64

			# Progress counters
			percent_period = ciso['total_blocks'] / 100
			percent_cnt = 0

			nocom_array = array.array('B')
			nocom_array.fromstring(struct.pack('{0}x'.format(ciso['total_blocks'])))
			for i in chain(*(xrange(a, b) for a,b in nocom_ranges)):
				try:
					nocom_array[i] = 1
				except IndexError:
					pass
			for i in chain(*(xrange(a, b) for a,b in vaccum_ranges)):
				try:
					nocom_array[i] = 2
				except IndexError:
					pass

			# print(len(nocom_array),sum(nocom_array))
			blocks = 0

			for block in range(0, ciso['total_blocks']):
				# Write alignment
				align = int(write_pos & align_m)
				if align:
					align = align_b - align
					size = fout.write(alignment_buffer[:align])
					write_pos += align
				
				# Mark offset index
				block_index[block] = write_pos >> ciso['align']

				# Read raw data
				raw_data = fin.read(ciso['block_size'])
				raw_data_size = len(raw_data)

				# Compress block
				# Compressed data will have the gzip header on it, we strip that.
				if nocom_array[block] == 1:
					writable_data = raw_data
					# Plain block marker
					block_index[block] |= 0x80000000
					# Next index
					write_pos += raw_data_size
				else:
					if nocom_array[block] == 2:
						# block += 1
						# print(block)
						raw_data = struct.pack('{0}x'.format(raw_data_size))
					compressed_data = zlib.compress(raw_data, compression_level)[2:]
					compressed_size = len(compressed_data)

					if compressed_size > (raw_data_size * COM_THRESHOLD):
						writable_data = raw_data
						# Plain block marker
						block_index[block] |= 0x80000000
						# Next index
						write_pos += raw_data_size
					else:
						writable_data = compressed_data
						# Next index
						write_pos += compressed_size

				# Write data
				fout.write(writable_data)

				# Progress bar
				percent = int(round((block / (ciso['total_blocks'] + 1)) * 100))
				if percent > percent_cnt:
					update_progress((block / (ciso['total_blocks'] + 1)))
					percent_cnt = percent

			# end for block
			# last position (total size)
			block_index[block+1] = write_pos >> ciso['align']

			# write header and index block
			print("Writing block index")
			fout.seek(CISO_HEADER_SIZE, os.SEEK_SET)
			write_block_index(fout, block_index)
		# end open(infile)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		sys.stderr.write('{0} isofile csofile\n'.format(os.path.basename(sys.argv[0])))
		raise SystemExit
	infile = sys.argv[1]
	outfile = sys.argv[2]


	iso = ISO9660(infile)
	# for i in iso.tree():
	# 	print(i)

	print('Scanning media files...')
	iso.scan_media_file()
	# mediatree = IntervalTree(iso._media_pos, 0, 1, 
	# 		min(i[0] for i in iso._media_pos), 
	# 		max(i[1] for i in iso._media_pos))
	# mediatree.pprint(2)
	for i in iso._media_file:
		print(i)

	# startlen = []
	# for i in vaccum_list:
	# 	s_l = iso.get_file_pos(i)
	# 	if s_l is not None:
	# 		startlen.append(s_l)
    #
	# if len(startlen):
	# 	with open(infile, 'r+b') as wfh:
	# 		for start, length in startlen:
	# 			wfh.seek(start)
	# 			for i in xrange(0, length/1024):
	# 				wfh.write(struct.pack('1024x'))
	# 			wfh.write(struct.pack('{0}x'.format(length % 1024)))

	compression_level = 9
	compress_iso(infile, outfile, compression_level, 
			iso._media_block, 
			iso._vaccum_block)

