import time
handleliste = []

for i in range(70000):
    handleliste.append("asdaslkjlkasdjlkjasldkjaslkdjalskjdasdlkjhei" + str(i))

varer = []

for i in range(4000000):
    if i%2:
        try:
            varer.append(handleliste[i])
        except:
            varer.append(":)")
    else:
        varer.append(":)")

varer = set(varer)
t = time.time()
counter = 0
for e in handleliste:
    if e in varer:
         counter += 1

print(counter)
print(round(time.time()-t, 2)) 