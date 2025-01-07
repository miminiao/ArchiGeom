import re
res=re.findall("(?<![A-Z0-9])[1-3]?AC(?![A-Z0-9])","A C")
print(res)