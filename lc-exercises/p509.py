class Solution(object):
	def fib(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		if n == 0: 
			return 0
		if n == 1:
			return 1
		dp1 = 0
		dp2 = 1
		res = 0
		for i in range(2, n+1):
			res = dp1 + dp2
			dp1 = dp2
			dp2 = res

		return res

  
s = Solution()
for i in range(10):
    print(s.fib(i))