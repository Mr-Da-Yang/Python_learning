'''
在terminal中输入python 3.py -h，查看
在terminal中输入python 3.py df --number1 4 -n2 7 赋值
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("parg")
parser.add_argument("-n1",'--number1',help='第一个参数',type=int)#"-n1"是'--number1'的简写
parser.add_argument("-n2",'--number2',help='第二个参数',type=int)
args=parser.parse_args()

print(args.parg)                      #在terminal中可直接python 3.py df，而不用写参数名，相反带有'-','--'的参数必须要加上参数名
print(args.number1)#调用时，只能写上全称，#在terminal中可以写全称或简写python 3.py df --number1 4 -n2 7
print(args.number2)
print(args)#显示所有参数Namespace(number1=5, number2=9, parg='sdf')


'''
\python_project\venv>python 3.py -h
usage: 3.py [-h] [-n1 NUMBER1] [-n2 NUMBER2] parg

positional arguments:
  parg

optional arguments:
  -h, --help            show this help message and exit
  -n1 NUMBER1, --number1 NUMBER1
                        第一个参数
  -n2 NUMBER2, --number2 NUMBER2
                        第二个参数
'''
