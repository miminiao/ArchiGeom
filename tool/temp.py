class A:
    def __init__(self,a:int,b:int) -> None:
        self.a=a
        self.b=b
class B:
    def __init__(self,a:int,b:int) -> None:
        self.a=A(a,b)
        self.b=b
b=B(1,2)
print(b.__dict__)