with open('meus-superpops.html','r',encoding='utf-8') as f:
    text = f.read()
start = text.index('<style>')
end = text.index('</style>', start)
block = text[start+7:end]
lines = block.strip('\n').split('\n')
for line in lines[-20:]:
    print(line)
