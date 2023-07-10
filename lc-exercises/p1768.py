class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        m, n = len(word1), len(word2)
        short = min(m, n)
        long = max(m, n)
        res = ''
        for i in range(short):
            res += word1[i]
            res += word2[i]

        if m < n :
            lw = word2
        else:
            lw = word1

        for i in range(short, long):
            res += lw[i]
        
        return res

  
if __name__ == '__main__':
    s = Solution()
    res = s.mergeAlternately('1111', '22222222')
    print(res)