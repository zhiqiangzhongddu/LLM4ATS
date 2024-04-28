from mordred import Calculator, descriptors

class OurMordredClass:
    calc = None
    
    def __init__(self, names: list[str]):
        self.calc = self.__filterMordredDisc(names)
    
    def get_unfiltered_calc(self):
        return Calculator(descriptors, ignore_3D=False)
    
    def __filterMordredDisc(self, names):
        dict = self.__discToIndexDict()
        lst1 = []
        for i in names:
            lst1.append(dict[i])
        calc = self.get_unfiltered_calc()
        lst = []
        for i in lst1:
            lst.append(calc.descriptors[i])
        calc.descriptors = lst
        return calc
    
    def __discToIndexDict(self):
        f = open("descriptors.txt", "r", encoding="utf8")
        props = f.read().split("\n\n")[1].split("\n") # select only among mordred descriptors
        propDict = {}
        for i in range(len(props)):
            propDict[props[i]] = i
        return propDict
    
    # Returns the list of descriptor values from a filtered descriptor list names
    def get_mordred_values(self,mols):
        return self.calc.pandas(mols).values.tolist()
    
    def get_specific_mordred_values(self, mols, names):
        return self.__filterMordredDisc(names).pandas(mols).values.tolist()


