from mordred import Calculator, descriptors

class OurMordredClass:
    all_props = None
    calc = None
    
    def __init__(self, select_props: list[str], all_props: list[str]):
        self.all_props = all_props
        self.calc = self.__filterMordredDisc(select_props)
    def get_unfiltered_calc(self):
        return Calculator(descriptors, ignore_3D=False)
    
    def __filterMordredDisc(self, names):
        dict = self.__discToIndexDict()
        lst1 = []
        for i in names:
            lst1.append(dict[i.lower()])
        calc = self.get_unfiltered_calc()
        lst = []
        for i in lst1:
            lst.append(calc.descriptors[i])
        calc.descriptors = lst
        return calc
    
    def __discToIndexDict(self):
        propDict = {}
        for i in range(len(self.all_props)):
            propDict[self.all_props[i]] = i
        return propDict
    
    # Returns the list of descriptor values from a filtered descriptor list names
    def get_mordred_values(self,mols):
        return self.calc.pandas(mols).values.tolist()
    
    def get_specific_mordred_values(self, mols, select_props):
        return self.__filterMordredDisc(select_props).pandas(mols).values.tolist()


