import os
import numpy as np
import cv2




#Python面向对象编程：

class Category(object):
#类属性，用于类的计数与标记    
    count=0

#调用静态方法（无参数，类名直接访问），如下：

    @staticmethod
    def show_help():
        
        print("显示了玩家帮助信息")
        
    
    def __init__(self,name):
        
        self.name=name
        self.data=10
        Category.count+=1
    
    def tensorflow_demo(self):
        
        #实例方法
        
        print("Hello")
        
    #调用类方法（参数为cls,cls.访问类属性，类名进行直接调用）：如下
    @classmethod
    def show_count(cls):
        
        print(cls.count)    
#单继承
class SmallCategory(Category):
    
    
    #继承初始化
    def __init__(self, name):
        super().__init__(name)
        self.data=100
    
    #重写
    def tensorflow_demo(self):
        return super().tensorflow_demo()
    
    def pytorch_demo(self):
        
        print("No Hello!")        

#多态
class OtherCategory(Category):
    def __init__(self, name):
        self.name=name
        self.data=1000
    def pytorch_demo(self):
        
        print("Other Hello")
   
#主函数
if __name__=="__main__":
    
    category=Category("老板")
    print(category.name)
    print(category.data)
    category.tensorflow_demo()
    category2=Category("老板娘")
    print(category2.name)
    
#访问类属性    
    print(Category.count)
#向上查找，先找方法内属性后找类属性，不推荐下面这种访问方法，推荐上面这种
    print(category2.count)
    
    Smallcategory=SmallCategory("家人")
    print(Smallcategory.name)
    print(Smallcategory.data)
    Smallcategory.tensorflow_demo()
    Smallcategory.pytorch_demo()
    
 #多态  
    other=OtherCategory("朋友")
    print(other.name)
    print(other.data)
    other.pytorch_demo()
    #多继承
    other.tensorflow_demo() 
    
#直接类调用：
    #调用类方法
    Category.show_count()
    #调用静态方法
    Category.show_help()

