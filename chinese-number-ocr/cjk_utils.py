import random

CJK_START = 0x4E00
CJK_END = 0x9FD5

numbers = '零一二三四五六七八九'

def random_cjk_character():
    '''Returns a character in the unified CJK range.'''
    return chr(random.randrange(CJK_START, CJK_END))

def random_cjk_number():
    '''Returns a single-digit number.'''
    return random.choice(numbers)

def enumerate_cjk_range(start=CJK_START, end=CJK_END, limit=None):
    if limit:
        end = start + limit
    for i in range(start, end):
        yield chr(i)


if (__name__ == '__main__'):
    print("Random CJK Character: ", random_cjk_character())
    print("Random CJK Number: ", random_cjk_number())
    print("CJK range, first 10 characters: ", 
          list(enumerate_cjk_range(limit=10)))
