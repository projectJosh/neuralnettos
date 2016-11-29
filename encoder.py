import numpy
from collections import OrderedDict


class encoder():
    #strings map to strings or floats, aka attribute category to values. Createa  function that takes a dataset, aka orderedDicts, and a list of attribute names, and
    #converts each orderedDict entry to a 2D array. Result should be a 3D array, use numpy.
    notUsed = []
    def __init__(self):
        self.notUsed.append(2)
        #self.values = OrderedDict()
    
    def decode(self, code, attributes):    
        vMap = {}
        for item in code:
            counter = 0
            for attribute in attributes:
                for value in self.values[attribute]:
                    if code[counter]==1:
                            vMap[attribute] = value
                    counter = counter +1
        return vMap
    
    def encode(self, listDic, attributes): #Given a dictionary of attributes->values, and the list of attributes, and then output a 2D array, 
            # in addition to the desired outputs for encoding, we also want to build a mapping that allows us to decode later. 
            # It would be <index of encoding, <attribute, value>>
            #dmap doesn't seem necessary at all, especially with decode() up above.
            
            #dmap = [] #decodeMap builds a list of <attribute, value> in the order that we build an encoding.
            #So, if we want to decode an encoding, we just create a counter at 0, iterate through for each attribute: for each value: and increment the counter at each value.
            #this list of dictionaries, which map attributes -> possible values of that attribute.
            #We need a list of all of the unique values across all attributes, as well as a mapping of attribute->list of possible values.
            #for instance in listDic:
            #    for attribute in attributes:
            #        if attribute not in self.values.keys():     
            #            self.values[attribute] = []
            #        if instance[attribute] not in self.values[attribute]:
            #            self.values[attribute].append(instance[attribute])
        
            resultA = [[] for l in listDic]
            resultL = [[] for l in listDic]
            #If attributes is an ordered Dictionary in which we have attributes -> list of possible values for that attribute:
            for i,instance in enumerate(listDic):
                for attribute in attributes.keys():
                    for v in attributes[attribute]: #We need a list of all the values for a given attribute.
                        if attribute == "label":
                            if instance[attribute] == v:
                                resultL[i].append(1)
                            else:
                                resultL[i].append(0)
                        else:
                            if instance[attribute] == v:
                                resultA[i].append(1)
                            else:
                                resultA[i].append(0)
            #Now we should have a 2d array of instances mapping to 1hotencoding of attributes for that instance.    
            #return [instances, 1hotencoding of attributes] and [instances, 1hotencoding of label]
            return resultA, resultL