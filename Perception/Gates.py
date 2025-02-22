#implementing different gate using the perception 


#Not gate
class PerceptronNOT:

    def __init__(self):
        self.w0 = 0.5
        self.w1 = -1

    def __call__(self, x):
        return self.forward(x)

    def decision_function(self, z):
        return 1 if z >= 0 else 0

    def forward(self, x1):
        z = self.w0 + self.w1 * x1
        phi_z = self.decision_function(z)
        return phi_z




#And Gate
class PerceptronAND:
    def __init__(self):
        self.w0 = -1.5
        self.w1 = 1    
        self.w2 = 1

    def __call__(self, x):
        return self.forward(x)

    
    def decision_function(self, z):
        return 1 if z >= 0 else 0
    
    def forward(self, x1, x2):
        z = self.w0 + self.w1 * x1 + self.w2 * x2
        phi_z = self.decision_function(z)
        return phi_z
    

class PerceptronOR:
    def __init__(self):
        self.w0 = -1
        self.w1 = 2    
        self.w2 = 2

    def __call__(self, x):
        return self.forward(x)

    
    def decision_function(self, z):
        return 1 if z >= 0 else 0
    
    def forward(self, x1, x2):
        z = self.w0 + self.w1 * x1 + self.w2 * x2
        phi_z = self.decision_function(z)
        return phi_z

class PerceptronNAND:
    def __init__(self):
        self.w0 = 1.5
        self.w1 = -1    
        self.w2 = -1

    def __call__(self, x):
        return self.forward(x)

    
    def decision_function(self, z):
        return 1 if z >= 0 else 0
    
    def forward(self, x1, x2):
        z = self.w0 + self.w1 * x1 + self.w2 * x2
        phi_z = self.decision_function(z)
        return phi_z
    

class PerceptronNOR:
    def __init__(self):
        self.w0 = 1
        self.w1 = -2    
        self.w2 = -2

    def __call__(self, x):
        return self.forward(x)

    
    def decision_function(self, z):
        return 1 if z >= 0 else 0
    
    def forward(self, x1, x2):
        z = self.w0 + self.w1 * x1 + self.w2 * x2
        phi_z = self.decision_function(z)
        return phi_z
    
class PerceptronMajorVote:
    def __init__(self):
        self.w0 = -1.5
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1

    def decision_function(self, z):
        return 1 if z >= 0 else 0

    def forward(self, x):
        # x is expected to be a tuple of three inputs (x1, x2, x3)
        x1, x2, x3 = x
        z = self.w0 + self.w1 * x1 + self.w2 * x2 + self.w3 * x3
        return self.decision_function(z)

    def __call__(self, x):
        return self.forward(x)


def main():
    model_NOT = PerceptronNOT()
    model_AND = PerceptronAND()
    model_OR = PerceptronOR()
    model_NAND = PerceptronNAND()
    model_NOR = PerceptronNOR()
    major_vote = PerceptronMajorVote()
    test = [(0, 0), (0, 1), (1, 0), (1, 1)]
    three_input_tests = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1)
    ]
    
    for x1 in [0, 1]:
        print(f"NOT({x1}) = {model_NOT(x1)}")
    
    for x1, x2 in test:
        print(f"AND({x1}, {x2}) = {model_AND(x1, x2)}")

    for x1, x2 in test:
        print(f"OR({x1}, {x2}) = {model_OR(x1, x2)}")
    
    for x1, x2 in test:
        print(f"NAND({x1}, {x2}) = {model_NAND(x1, x2)}")

    for x1, x2 in test:
        print(f"NOR({x1}, {x2}) = {model_NOR(x1, x2)}")

    for inp in three_input_tests:
        print(f"MajorVote{inp} = {major_vote(inp)}")




if __name__ == '__main__':
    main()