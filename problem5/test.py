def RSHash_cpu(strn, s):

	b = 378551
	a = 63689
	h = 0
	for i in range(0, s):

		h = h * a + ord(strn[i])
		a    = a * b

	return (h & 0x7FFFFFFF)


print(RSHash_cpu("e2gxgd\0", 7))
print(RSHash_cpu("paral12", 7))

