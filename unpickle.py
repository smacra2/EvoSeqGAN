import pickle

unpickled_store = []
file_id = open('cleanData', 'rb')
while True:
    try:
        unpickled_item = pickle.load(file_id)
        unpickled_store.append(unpickled_item)
    except EOFError:
        break
file_id.close()

f = open('cleanData.txt', 'w')
f.write(str(unpickled_store))
f.close()
