class A:
    def __init__(self):
        print("A.init")
        self.f()
    def f(self):
        print("A.f")
    def g(self):
        print("A.g")
class B(A):
    def __init__(self):
        super().__init__()
        print("B.init")
        self.f()
    def f(self):
        print("B.f")
a=A()
b=B()
...