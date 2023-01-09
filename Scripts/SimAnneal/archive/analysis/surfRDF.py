blankIdx, numAtoms = [], []
with open("/g/data/q27/jt5911/NCPac/ov_SURF_layer.xyz", "r") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        if len(line) == 1:
            blankIdx.append(i)
print(len(blankIdx))

for (i, idx) in enumerate(blankIdx):
    if i != 0:
        numAtom = blankIdx[i] - blankIdx[i-1] - 2
        numAtoms.append(numAtom)
print(len(numAtoms))

j = 0
with open("/g/data/q27/jt5911/NCPac/testing.xyz", "w") as f:
    for (i, line) in enumerate(lines):
        if i == len(lines) - 1:
            break
        if j >= len(numAtoms):
            #f.write(line)
            newLine = " ".join(line.split())
            f.write(newLine + "\n")
            continue
        if i == blankIdx[j] - 1:
            f.write(str(numAtoms[j]) + "\n")
            j += 1
        else:
            newLine = " ".join(line.split())
            f.write(newLine + "\n")
print("Done! Check the element number of the last frame!")
