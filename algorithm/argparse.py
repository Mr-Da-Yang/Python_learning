'''
在terminal中输入python 文件名.py -h，查看
在terminal中输入python 3.py df --number1 4 -n2 7 赋值#df是给默认参数‘parg’赋值的
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("parg")
parser.add_argument('--dataset_path', type=str, default='./data')
parser.add_argument("-n1",'--number1',help='第一个参数',type=int)#"-n1"是'--number1'的简写
parser.add_argument("-n2",'--number2',help='第二个参数',type=int)

train(args):
    print(args.dataset_path)
      
 #1   
args=parser.parse_args()
print(args.parg)                      #在terminal中可直接python 3.py df，而不用写参数名，相反带有'-','--'的参数必须要加上参数名
print(args.number1)#调用时，只能写上全称，#在terminal中可以写全称或简写python 3.py df --number1 4 -n2 7
print(args.number2)
print(args)#显示所有参数Namespace(dataset_path='./data'，number1=5, number2=9, parg='sdf')按A-Z顺序出现

    
#2 供其他函数调用    
args=parser.parse_args()
a=train(args)



#3在ternimal中输入 python 文件名.py -h  查看
'''
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

#4
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument("-n2",'--number2',help='第二个参数',type=int)
#1.第一个的action='store_true'，保证输出的是True or False，如果没有这个action的话，输出None，且，这里print出的是False
#没有default的话，action='store_true'是False，action='store_false'是True
#所以加上默认值来限制parser.add_argument('--usecpu', action='store_false',default='Ture', help="use cpu")这时的action='store_false'，不起作用

#2.action='store_true'终端调用就为True
#对于上面的--use-cpu， run py时就是上述1描述的情况，如果terminal的话，action='store_true'是False，变成了ture

#3.对于n2来讲，没有default的默，认值，所以print（args.number2）=None，需要自己填写
