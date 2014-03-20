import itertools
from operator import itemgetter

def combiner(big_set,count):
    subsets = list(itertools.combinations(sorted(big_set), count))
    if count == 1:
        subsets = [item for sublist in subsets for item in sublist]
    return subsets

def generate_freq_sets(file,levels, support):
    #print levels, support
    freq_items_list = []    
    for  i in range(levels):
        k = i + 1
   #     counter = 0 
        freq_items_list.append({})
        #print i
        with open(file,'r') as f2:
            for line in f2:
                items = line.split()
                combine = combiner(items,k)            
                for itemset in combine:
                    frequent  =  True
                    ##############  check if subsets are frequent ########
                    if k >= 2 :
                        subset = combiner(itemset,k-1)
                        for s in subset:
                            if s not in freq_items_list[k-2]:
                                frequent = False
                    if frequent == True:
                        if itemset in freq_items_list[k-1]:
                            freq_items_list[k-1][itemset] = freq_items_list[k-1].get(itemset) + 1
                        else:
                            freq_items_list[k-1][itemset] = 1                
        f2.close()
        ###### Pruning #############
        for key in  freq_items_list[k-1].keys():
            if freq_items_list[k-1][key] < support:
                del freq_items_list[k-1][key]

    return freq_items_list


file = '/Users/sdey/Documents/cs246/home-work/hw01/browsing.txt'

freq_items  = generate_freq_sets(file, 3, 100)

asso_rules_1 = {}

for key in freq_items[1].keys():
     subsets = combiner(key,1)
     for s in subsets:
         rest = tuple([val for val in key if val not in s])[0]
         k =  (rest,s)  
        # print k
         confidence = float(freq_items[1][key])/ freq_items[0][rest]
    #     print confidenceprint
         asso_rules_1[k]=confidence

sorted_list_2 = sorted(asso_rules_1.items(), key=itemgetter(1,0),reverse=True)

for i in sorted_list_2[0:5]:
    print i[0][0], '->',i[0][1], ', Confidence:',i[1]

asso_rules_2 = {}

for key in freq_items[2].keys():
     subsets = combiner(key,1)
     for s in subsets:
         t = tuple(sorted([val for val in key if val not in s]))
         rest = (t[0],t[1])
         k =  (t[0],t[1],s)
        # print key,'s=',s,'t=',t,'rest=',rest
         confidence = float(freq_items[2][key])/ freq_items[1][rest]
        # print confidence
         asso_rules_2[k]=confidence
         
sorted_list_3 = sorted(asso_rules_2.items(), key=itemgetter(1,0),reverse=True)

for i in sorted_list_3[0:5]:
    print i

for i in sorted_list_3[0:5]:
    print i[0][0], ',',i[0][1],'->',i[0][2], ', Confidence:',i[1]


