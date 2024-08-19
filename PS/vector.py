
import math as m

class Vec4:

    def __init__(self,px=0,py=0,pz=0,E=0):
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz

    def __getitem__(self,i):
        if i == 3: return self.E
        if i == 0: return self.px
        if i == 1: return self.py
        if i == 2: return self.pz
        raise Exception('Vec4D')

    def __repr__(self):
        return '({0},{1},{2},{3})'.format(self.px,self.py,self.pz,self.E)

    def __str__(self):
        return '({0},{1},{2},{3})'.format(self.px,self.py,self.pz,self.E)

    def __add__(self,v):
        return Vec4(self.px+v.px,self.py+v.py,self.pz+v.pz,self.E+v.E)

    def __sub__(self,v):
        return Vec4(self.px-v.px,self.py-v.py,self.pz-v.pz,self.E-v.E)

    def __neg__(self):
        return Vec4(-self.px,-self.py,-self.pz,-self.E)

    def __mul__(self,v):
        if isinstance(v,Vec4):
            return self.E*v.E-self.px*v.px-self.py*v.py-self.pz*v.pz
        return Vec4(self.px*v,self.py*v,self.pz*v,self.E*v)

    def __rmul__(self,v):
        if isinstance(v,Vec4):
            return self.E*v.E-self.px*v.px-self.py*v.py-self.pz*v.pz
        return Vec4(self.px*v,self.py*v,self.pz*v,self.E*v)

    def __div__(self,v):
        return Vec4(self.px/v,self.py/v,self.pz/v,self.E/v)

    def M2(self):
        return self*self

    def M(self):
        return m.sqrt(self.M2())

    def P2(self):
        return self.px*self.px+self.py*self.py+self.pz*self.pz

    def P(self):
        return m.sqrt(self.P2())

    def PT2(self):
        return self.px*self.px+self.py*self.py

    def PT(self):
        return m.sqrt(self.PT2())

    def Theta(self) :
        return m.acos(self.pz/self.P())

    def Phi(self) :
        if self.px==0 and self.py==0:
            return 0.0
        else:
            return m.atan2(self.py,self.px)

    def Cross(self,v):
        return Vec4(self.py*v.pz-self.pz*v.py,
                    self.pz*v.px-self.px*v.pz,
                    self.px*v.py-self.py*v.px,
                    0.0)

    def Boost(self,v):
        rsq = self.M()
        v0 = (self.E*v.E-self.px*v.px-self.py*v.py-self.pz*v.pz)/rsq;
        c1 = (v.E+v0)/(rsq+self.E)
        return Vec4(v.px-c1*self.px,
                    v.py-c1*self.py,
                    v.pz-c1*self.pz,
                    v0)

    def BoostBack(self,v):
        rsq = self.M()
        v0 = (self.E*v.E+self.px*v.px+self.py*v.py+self.pz*v.pz)/rsq;
        c1 = (v.E+v0)/(rsq+self.E)
        return Vec4(v.px+c1*self.px,
                    v.py+c1*self.py,
                    v.pz+c1*self.pz,
                    v0)
