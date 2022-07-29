#!/usr/bin/env python

import copy
import string
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from multiprocessing import Pool
import os
import itertools
import tqdm

class Polynom:

	def __init__(self, s, P):
		self.P = P

		if isinstance(s, list):
			n = len(s)
			if n > 0:
				assert s[n-1] != 0

			self.coeffs = s
		elif isinstance(s, str):
			xs = s.split(" + ")
			xs = list(map(lambda x: x.split("x^"),xs))
			# print(s)
			# print(xs)

			self.coeffs = [0] * (int(xs[len(xs)-1][1]) + 1)
			i = 0
			for x in xs:
				power = int(x[1])
				a = int(x[0])

				self.coeffs[power] = a

	def degree(self):
		return len(self.coeffs) - 1


	def __getitem__(self, i):
		return self.coeffs[i] if i <= self.degree() else 0

	# ~ O(max(degree(self),degree(other)))
	def __add__(self, other):
		n = max(self.degree(), other.degree())
		new_coeffs = []

		# start at highest degree
		for i in range(n,-1,-1):
			c = (self[i] + other[i]) % self.P
			if c == 0 and len(new_coeffs) == 0:
				continue
			else:
				new_coeffs.append(c)

		return Polynom(list(reversed(new_coeffs)), self.P)

	def __mul__(self, other):
		if isinstance(other, self.__class__):
			return self.polynom_mul(other)
		elif isinstance(other, int):
			return self.scale(other)

	# ~ O((degree(self)+degree(other))^2)
	def polynom_mul(self, other):
		a = self.degree()
		b = other.degree()

		if a == -1 or b == -1:
			return Polynom([], self.P)

		n = a + b

		new_coeffs = []

		for i in range(0,(n+1)):
			c = 0
			for k in range(0, i+1):
				c += self[k]*other[i-k]

			new_coeffs.append(c % self.P)

		return Polynom(new_coeffs, self.P)

	def scale(self, scalar):
		for i in range(0, len(self.coeffs)):
			self.coeffs[i] = (self.coeffs[i] * scalar) % self.P
		return self

	# ~ O((n-m)*m)
	# where n = degree(self), m = degree(other)
	def __mod__(self, other):
		# print(self, "%", other)

		f = copy.deepcopy(self)

		P = self.P
		n = f.degree()
		m = other.degree()

		b = inv(other[m],P)

		# cancel highest coeff as long as m <= n
		while n >= m:
			a = f[n]
			new_n = n
			uneq0 = False
			for i in range(n,n-m-1, -1):
				# print("    in:", f.coeffs[i], - a*b*other[i-(n-m)])
				f.coeffs[i] = (f.coeffs[i] - a*b*other[i-(n-m)]) % P
				# print("    out:", f.coeffs[i])

				if f.coeffs[i] == 0 and not uneq0:
					new_n = new_n-1
				else:
					uneq0 = True # as long as the coeff is 0 we decrease the degree
			n = new_n
			# print("    g:", f)

		return f.prune()

	def is_divisible_by(self, other):
		return (self % other).degree() == -1


	# removes 0s
	def prune(self):
		n = self.degree()
		if n == -1:
			return self

		while self[n] == 0 and n >= 0:
			n = n - 1

		new_coeffs = []
		for i in range(0, n+1):
			new_coeffs.append(self[i])

		self.coeffs = new_coeffs

		return self

	def __str__(self):

		if self.degree() == -1:
			return "0"

		s = ""
		add = False
		for (i, a) in enumerate(self.coeffs):
			if a != 0 and add == False:
				if i == 0:
					s += f"{a}"
				elif i == 1 and a == 1:
					s += f"x"
				elif i == 1:
					s += f"{a}x"
				elif a == 1:
					s += f"x^{i}"
				else:
					s += f"{a}x^{i}"
				add = True
				continue

			if add == True:
				if a == 0:
					continue
				elif i == 1 and a == 1:
					s += f" + x"
				elif i == 1:
					s += f" + {a}x"
				elif a == 1:
					s += f" + x^{i}"
				else:
					s += f" + {a}x^{i}"
		return s

	def __repr__(self):
		return str(self)

	def __hash__(self):
		n = len(str(self.P))
		s = ""
		for a in reversed(self.coeffs):
			x = str(a)
			while len(x) < n: x = '0' + x
			s += x

		if s == "":
			return 0

		return int(s)

	def __eq__(self, other):
		return hash(self) == hash(other)


def inv(x, mod):
	for y in range(1, mod):
		if (x * y) % mod == 1:
			return y

def devisors(x):
	d = []
	for y in range(1,x):
		if (x % y) == 0:
			d.append(y)
	return d

def greatest_divisor(x):
	d = x - 1
	while (x % d) != 0:
		d = d -1
	return d

class Field:

	def __init__(self, N, P, cache_tables):
		self.N = N
		self.P = P

		self.divisors = devisors(N)
		self.elems = []
		self.symbols = dict()
		self.numbers = dict()
		n = pow(P,N)

		self.addition = np.zeros((n,n))
		self.multiplication = np.zeros((n,n))
		self.orders = np.zeros(n)

		t0 = time.time()

		tables_loaded = False

		if cache_tables:
			try:
				self.load_tables_from_cache()
				tables_loaded = True
			except:
				pass


		self.find_field()

		t1 = time.time()
		print(f"found polynomial: {t1-t0}s")

		if not tables_loaded:
			self.calculate_tables(multi_thread=False)
			t2 = time.time()
			print(f"calculated tables: {t2-t1}")

		self.calculate_orders()
		self.find_primitve_element()

		t3 = time.time()
		print(f"total time: {t3-t0}s")

		if not tables_loaded and cache_tables:
			self.cache_tables()

		self.plot()


	def find_field(self):
		# has to be of degree n
		# has to divide x^(p^n)-x
		# factors of x^(p^n)-x can only be divided by polynoms of degree k | n
		N = self.N
		P = self.P

		coeffs = [0] * (N+1)
		coeffs[N] = 1

		q = Polynom(f"-1x^1 + 1x^{pow(P,N)}", P)
		f = Polynom(coeffs, P)

		self.find_irreducible_polynom(N-1, q, f)
		print("f", f)

		hs = [0] * N
		hs[N-1] = 1
		h = Polynom(hs, P)
		print("h", h)

		# DO NOT ALTER ONCE SET
		# ENUMERATE IS HANDY
		# numbers[elems[i]] = i
		self.find_all_normed_polynoms_of_degree(N-1, h, self.elems)

		# for elem in self.elems:
		# 	print(elem)

	# ~ O((p^n)(p^n+1)/2 * 6n^2) = O(3 * p^(2n) * n^2)
	def calculate_tables(self, multi_thread=True):
		for (i,x) in enumerate(self.elems):
			self.numbers[x] = i

		if multi_thread:
			pool = Pool(os.cpu_count())
			domain = itertools.product(enumerate(self.elems), enumerate(self.elems))

			t0 = time.time()

			def predicate(t):
				(i,x), (j,y) = t
				return i >= j

			params = list(filter(predicate, domain))

			t1 = time.time()
			print(f"parameter generated: {t1-t0}s")

			total = len(params)

			print("calculating operations:")
			# res = pool.map(self.calculate, params)
			chunksize = self.calc_chunksize(os.cpu_count(),total)
			res = list(tqdm.tqdm(pool.imap_unordered(self.calculate, params, chunksize=chunksize), total=total))

			pool.close()

			for (i,j,rn,sn) in res:
				self.multiplication[i][j] = rn
				self.multiplication[j][i] = rn
				self.addition[i][j] = sn
				self.addition[j][i] = sn



		else:
			count = 0
			total = int(len(self.elems) * (len(self.elems)+1)  / 2)
			# t0 = time.time()
			for (i,x) in enumerate(self.elems):
				eta = None
				for (j,y) in enumerate(self.elems):
					if i < j:
						continue
					(i,j,rn,sn) = self.calculate(((i,x),(j,y)))

					self.multiplication[i][j] = rn
					self.multiplication[j][i] = rn
					self.addition[i][j] = sn
					self.addition[j][i] = sn

					count += 1
					self.print_progress(count, total)

	def calc_chunksize(self, n_workers, len_iterable, factor=4):
	    chunksize, extra = divmod(len_iterable, n_workers * factor)
	    if extra:
	        chunksize += 1
	    return chunksize

	def calculate(self, params):
		(i,x), (j,y) = params

		r = (x * y) % self.f # ~ O((2n)^2) + O((2n-n)*n) = O(5n^2)
		s = (x + y) % self.f # ~ O(n) + O((2n-n)*n) = O(n^2)
		#print("(", x, ")", "*", "(", y, ")", "=", r)
		#print("(", x, ")", "+", "(", y, ")", "=", s)

		rn = self.numbers[r]
		sn = self.numbers[s]

		# print(i,j,rn,sn, self)
		return (i,j,rn,sn)

	def calculate_multi(self, params):
		tmp0 = np.ctypeslib.as_array(self.shared_array0)
		tmp1 = np.ctypeslib.as_array(self.shared_array1)

		(i,x), (j,y) = params

		r = (x * y) % self.f # ~ O((2n)^2) + O((2n-n)*n) = O(5n^2)
		s = (x + y) % self.f # ~ O(n) + O((2n-n)*n) = O(n^2)


		rn = self.numbers[r]
		sn = self.numbers[s]

		tmp0[i][j] = rn
		tmp0[j][i] = rn
		tmp1[i][j] = sn
		tmp1[j][i] = sn

	def print_progress(self, count, total):
		# t = time.time() - t0
		progress = round(100* count/total)
		# eta = round(t / (progress+0.001) * 100 - t)
		sys.stderr.write(f"{count}/{total} ({progress}%) operations done...\r")

	def cache_tables(self):
		np.savetxt(f"cache/{self.P}^{self.N}_add.txt", self.addition, fmt='%i')
		np.savetxt(f"cache/{self.P}^{self.N}_mul.txt", self.multiplication, fmt='%i')
		# np.savetxt(f"cache/{self.P}^{self.N}_ord.txt", self.orders, fmt='%i')

	def load_tables_from_cache(self):
		self.addition = np.loadtxt(f"cache/{P}^{N}_add.txt")
		self.multiplication = np.loadtxt(f"cache/{P}^{N}_mul.txt")
		# self.orders = np.loadtxt(f"cache/{P}^{N}_ord.txt")

	def plot(self, labels = False):
		P = self.P
		N = self.N

		fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True)

		s = r"$F_{" + str(pow(P,N)) + r"} = F_{" + f"{P}^{N}"
		s += r"}\cong \mathbb{Z}_{" + str(P) + r"}[x] / (" + str(self.f)
		s +=  r") = \langle " +  str(self.primitive_elem) + r" \rangle $"

		fig.suptitle(s, fontsize=20, fontweight='bold', y=0.985)

		# ax0 = fig.add_subplot(gj[0])
		# ax1 = fig.add_subplot(gj[1])

		ax0.matshow(self.addition, cmap=plt.cm.Blues) #
		ax1.matshow(self.multiplication, cmap=plt.cm.Blues)

		ax0.set_title(r"$(F_{" + str(pow(P,N)) + r"},+) \cong C_{" + str(P) + r"}^" + str(N) + r"$",y=1.1)
		ax1.set_title(r"$(F_{" + str(pow(P,N)) + r"}  \backslash \{ 0 \},*) \cong C_{" + str(pow(P,N)-1) + r"}$", y=1.1)

		# for (i,x) in enumerate(self.elems):
		# 	print(x, ",\thash:", hash(x), ",\tnum:", self.numbers[x])

		n = pow(P,N)
		if labels:
			for i in range(0, n):
				for j in range(0, n):
					a = self.addition[i][j]
					m = self.multiplication[i][j]
					ax0.text(i, j, str(int(a)), va='center', ha='center')
					ax1.text(i, j, str(int(m)), va='center', ha='center')

		plt.show()


	def print_symbol_tables(self):
		abc = list(string.ascii_lowercase)

		for (i,x) in enumerate(self.elems, -2):
			if str(x) == "0":
				self.symbols[x] = "0"
			elif str(x) == "1":
				self.symbols[x] = "1"
			else:
				self.symbols[x] = abc[i]


		header = "\t"

		for x in self.elems:
			header += self.symbols[x] + "\t"

		mtable = header + "\n\n"
		atable = header + "\n\n"

		for x in self.elems:
			mrow = self.symbols[x] + "\t"
			arow = self.symbols[x] + "\t"
			for y in self.elems:
				r = (x * y) % self.f
				s = (x + y) % self.f
				mrow += self.symbols[r] + "\t"
				arow += self.symbols[s] + "\t"

			mtable += mrow + "\n"
			atable += arow + "\n"

		P = self.P
		N = self.N

		print(f"\nG({P}^{N}) ~ Z_{P} / ({self.f})")

		print("\nADDITION:\n")
		print(atable)

		print("\nMULTIPLICATION:\n")
		print(mtable)

	# f has to be of degree d, permutates 0 to d coeff
	def find_all_normed_polynoms_of_degree(self, d, f, a):
		P = self.P
		if d == 0:
			for k in range(0, P):
				f.coeffs[d] = k
				a.append(copy.deepcopy(f).prune())

		else:
			for k in range(0, P):
				f.coeffs[d] = k
				self.find_all_normed_polynoms_of_degree(d-1, f, a)



	# all normed polynoms f of degree n are checked ~ p^n
	# check if x^(p^n)-x is divisible by f ~ O((p^n-n)*n)
	# check if f is irreducible, all normed polynoms of degree d | n are checked ~ sum(p^d*(n-d)*d: d|n)
	# theorethical worst case ~ O(p^n * p^n * n * n/2 * p^(n/2) * n^2 /2)
	# but much faster in reality
	def find_irreducible_polynom(self, i, q, f):
		P = self.P
		if i == 0:
			for k in range(0, P):
				f.coeffs[i] = k
				if q.is_divisible_by(f):
					if self.is_irreducible(f):
						self.f = f
						return True

		else:
			for k in range(0, P):
				f.coeffs[i] = k
				if self.find_irreducible_polynom(i-1, q, f):
					return True


	def is_irreducible(self, f):
		for d in self.divisors:
			hs = [0] * (d + 1)
			hs[d] = 1 # d degree normed
			h = Polynom(hs, self.P)
			all_polys = []
			self.find_all_normed_polynoms_of_degree(d-1, h, all_polys)

			for g in all_polys:
				if f.is_divisible_by(g):
					return False

		return True

	def find_primitve_element(self):
		elem = None
		order = 0
		for (i,x) in enumerate(self.elems):
			o = self.orders[i]
			if o > order:
				elem = x
				order = o

		# print(f"primitive element: {i} = {elem}")
		self.primitive_elem = elem

		# j = i
		# for n in range(2, len(self.elems)):
		# 	j = int(self.multiplication[j][i])
		# 	print(f"({elem})^{n}\t=\t{self.elems[j]}\t\t=\t{j}")



	def calculate_orders(self):
		pool = Pool(os.cpu_count())
		res = pool.map(self.calculate_order, enumerate(self.elems))
		pool.close()

		for (x,n) in res:
			self.orders[x] = n


	def calculate_order(self, params):
		(i,x) = params
		if hash(x) == 1:
			return (i,1)

		y = i
		for n in range(2,len(self.elems)):
			y = int(self.multiplication[y][i])
			if y == 1:
				return (i, n)
		return (i, 0)


if __name__ == '__main__':

	P = int(input("prime number: "))
	N = int(input("natural number: "))
	#b = bool(input("use cache: "))
	b = False
	Field(N=N,P=P,cache_tables=b)
