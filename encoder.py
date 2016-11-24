import numpy



class encoder():
    #strings map to strings or floats, aka attribute category to values. Createa  function that takes a dataset, aka orderedDicts, and a list of attribute names, and
    #converts each orderedDict entry to a 2D array. Result should be a 3D array, use numpy.
    uniqueValues = []
    values = []
    
    def decode(self, code, attributes):    #We can send this method the 1hotencoding and it should return the values associated with it.
        vMap = []
        for item in code:
            counter = 0
            for attribute in attributes:
                for value in self.values(attribute):
                    if(code[counter]==1):
                            vMap.append(attribute)
                            vMap[attribute].append(value)
                    counter = counter +1
    
    def skconvert(self, listDic, attributes): #Given a dictionary of attributes->values, and the list of attributes, and then output a 2D array, 
            # in addition to the desired outputs for encoding, we also want to build a mapping that allows us to decode later. 
            # It would be <index of encoding, <attribute, value>>
            
            #dmap doesn't seem necessary at all, especially with decode() up above.
            
            dmap = [] #decodeMap builds a list of <attribute, value> in the order that we build an encoding.
            #So, if we want to decode an encoding, we just create a counter at 0, iterate through for each attribute: for each value: and increment the counter at each value.
            #this list of dictionaries, which map attributes -> possible values of that attribute.
            #We need a list of all of the unique values across all attributes, as well as a mapping of attribute->list of possible values.
            for instance in listDic:
                for attribute in attributes:
                    if attribute not in self.values:     
                        self.values.append(attribute)
                    if instance[attribute] not in self.values[attribute]:
                        self.values[attribute].append(instance[attribute])
                        self.uniqueValues.append(instance[attribute])
        
            resultA = numpy.zeros(len(listDic),len(self.uniqueValues))
            resultL = numpy.zeros(len(listDic),len(self.uniqueValues))
        
            #If, for each instance, we iterate through attribute in attributes, check the index of each attribute in attributes, and then check that index in the instances'
            # list of values, then that might work? 
            #I don't see how to do this, we need to have subsets of the uniqueValues, so that it's categorized by attribute.
            for attribute in attributes:
                dmap.append(attribute)
                for value in self.values(attribute):
                    dmap[attribute].append(value)
                    
            for instance in listDic:
                for attribute in attributes:
                    for v in self.values[attribute]: #We need a list of all the values for a given attribute.
                        if attribute == "label":
                            for label in self.values["label"]:
                                if label == v:
                                    resultL[instance].append(1)   
                        else:
                            if instance[attribute] == v:
                                resultA[instance].append(1)
            #Now we should have a 2d array of instances mapping to 1hotencoding of attributes for that instance.    
            #return [instances, 1hotencoding of attributes] and [instances, 1hotencoding of label]
            return resultA, resultL