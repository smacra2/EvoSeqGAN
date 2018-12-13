import pickle
import numpy as np



def mycode(ref):
    
        tmp=np.zeros((len(ref),5), dtype=np.bool)
        # encode refrence character
        for j in range(0,len(ref)):
            if ref[j]=='A':
                tmp[j,0]=1
            elif ref[j]=='C':
                tmp[j,1]=1
            elif ref[j]=='G':
                tmp[j,2]=1
            elif ref[j]=='T':
                tmp[j,3]=1
            else:
                tmp[j,4]=1
            
        return tmp


def decode(ref):
        dict='ACGT-'    
        cod=''
        for j in range(0,ref.shape[0]):
            ind=np.argmax(ref[j,:])
            cod+=dict[ind]            
        return cod


def mydecode(ref):
        dict='ACGT-'    
        cod=''
        for j in range(0,ref.shape[0]):
            ind=np.where(ref[j,:]==1)[0][0]
            cod+=dict[ind]            
        return cod
def mydecodepred(ref):
        dict='ACGT-'    
        cod=''
        for j in range(0,ref.shape[0]):            
            cod+=dict[ref[j]]            
        return cod



def readmatches(fname,tag1,tag2):
    # this function reads the file and find reference sequence starts by tag1 and candidate sequence starts by tag2

    # read all lines
    f = open(fname)
    txt = f.readlines()
    f.close()    

    ref=[]
    refind=[]
    real=[]
    realind=[]
    # find lines includes tag1
    for i in range(0,len(txt)): 
        tmp=txt[i]
        if tmp[0:len(tag1)]==tag1: # if found
            # add sequence into reference list and keep the sequence line number too
            x=tmp.split()
            ref.append(x[-1].upper())
            refind.append(i)

    # find lines includes tag2
    for i in range(0,len(txt)): 
    
         tmp=txt[i]
         if tmp[0:len(tag2)]==tag2: # if found
             # add sequence into candidate list and keep the sequence line number too
             x=tmp.split()
             real.append(x[-1].upper())
             realind.append(i)       
   
    # sometimes there is reference sequences but ther eis no candidate sequence. So we need to match read sequence
    matches=[]
    refind.append(len(txt))

    i=0;j=0;

    # we expected every reference sequence followed by candidate sequence. if two reference sequence comes 
    # but there is no candidate sequence neglect read reference sequence

    while i<len(ref) and j< len(real):
        if realind[j]>refind[i] and realind[j]<refind[i+1]:
            # if corresponding reference and canidadtae sequecne are matched. then added them into matches list
            matches.append((ref[i],real[j]))
            i+=1
            j+=1
        elif realind[j]<refind[i]:
            j+=1
        else:
            i+=1
    # there are two pair of sequnece so return that list
    return matches


def prepareData(realmatches,yhat):
    # this function for encoding the character sequence by numbers.
    # we used binary coding. A:10000, C:01000, G:00100, T:00010, _:00001
    X=[]
    Y=[]
    # for each mathces
    for i in range(0,len(realmatches)):
        ref=realmatches[i][0]
        cnd=realmatches[i][1]
        # prepare encoding sequence
        tmp=np.zeros((1,len(ref),10))
        # encode refrence character
        for j in range(0,len(ref)):
            if ref[j]=='A':
                tmp[0,j,0]=1
            elif ref[j]=='C':
                tmp[0,j,1]=1
            elif ref[j]=='G':
                tmp[0,j,2]=1
            elif ref[j]=='T':
                tmp[0,j,3]=1
            else:
                tmp[0,j,4]=1
            # encode candidate character
            if cnd[j]=='A':
                tmp[0,j,5]=1
            elif cnd[j]=='C':
                tmp[0,j,6]=1
            elif cnd[j]=='G':
                tmp[0,j,7]=1
            elif cnd[j]=='T':
                tmp[0,j,8]=1
            else:
                tmp[0,j,9]=1
        # add encoded sequence pair and its label
        X.append(tmp)
        Y.append([yhat])
    # return encoded sequence and its label
    return X,Y


if __name__ == "__main__": 

    # read sequences from certain files which starts by given tag1 and tag2
    realmatches=readmatches('Real_Alignments_1.000.000_Lines.txt','s hg38','s _HPG ')
    fakematches=readmatches('Fake_Alignments_100.000_Lines.txt','s Human.','s _HR ')

    # make the character sequence into number by coding
    Xt,Yt=prepareData(realmatches,1)
    Xf,Yf=prepareData(fakematches,0)
    
    # reorder the fake and real sequence randomly
    p=np.random.permutation(len(realmatches)+len(fakematches))
    
    X=[]
    Y=[]
    
    for i in range(0,len(realmatches)+len(fakematches)):
        if p[i]<len(realmatches):
            X.append(Xt[p[i]])
            Y.append(Yt[p[i]])
        else:
            X.append(Xf[p[i]-len(realmatches)])
            Y.append(Yf[p[i]-len(realmatches)])

    # write reordered data randomly into the pickle file
    pickle_out = open('cleandata',"wb")
    pickle.dump(realmatches, pickle_out)        
    pickle.dump(fakematches, pickle_out)   
    pickle.dump(X, pickle_out) 
    pickle.dump(Y, pickle_out)     
    pickle_out.close()
    



