# #######################
# Author: Kyle Gao 
# Sept 2020
# @Sheffield, UK 

# #######################


try:
    os.makedirs( os.path.join(os.getcwd(), 'folderName') )
except OSError:
    pass

for f in os.listdir('frames'):


users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
users.show()
userSubsetRecs.show(3,False)

movieRecs = model.recommendForAllItems(10)
movieRecs.show(5, False)


wget <url>  # defalut dl to current path.
wget -P <directory> <url> 

----------Dissertation  configuration------------------------------------------
# For GPU, you can start an interactive session on ShARC for coding and debugging. 
qrshx -l gpu=1 

# For torchvision, you need to install torchvision and pytorch in your own anaconda virtual environment.
# If you want to install torch 1.4.0 and torchvision 0.5.0 for Cuda 100, try this in your own env:

pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
---------------------------------------------------------------



# unzip zip file
unzip epic-download-scripts-master.zip -d ./
# mv download-scripts-master/* ../
# sh download-scripts-master/videos/test/download_P16.sh ~/Desktop
unzip ClaimPredictionChallenge.zip -d ./ClaimPredictionChallenge/
unzip train_set.zip -d ./train_set/


brew install wget


# dwon load the raw video 
sh /fastdata/acq18jg/download-scripts-master/download_P1_8_22.sh /fastdata/acq18jg/
sh /fastdata/acq18jg/download-scripts-master/download_rgb_1_8_22.sh /fastdata/acq18jg/
sh /fastdata/acq18jg/download-scripts-master/download_flow_1_8_22.sh /fastdata/acq18jg/
#***  du -h your_directory # to show the size of directory
#***  du -smh filename.zip
#***  df -h dir_name # to check the available capacity!! 
#***  du -h -d1 ./   # only the sub-folder demonstrated 

#  env  
#  ref: https://github.com/epic-kitchens/starter-kit-action-recognition
cd /data/acq18jg 
$ git clone https://github.com/epic-kitchens/starter-kit-action-recognition.git epic
$ cd epic

conda create -n epic-g1 python=3.6 ipykernel jupyter_client
source activate epic-g1


$ conda env create -n gg-epic -f environment.yaml
# $ conda activate gg-epic
$ source activate gg-epic
pip install --upgrade pip
pip install epic-kitchens
conda install -c conda-forge numpy
conda install -c conda-forge pandas 
conda install -c conda-forge scipy
conda install -c conda-forge pytorch
conda install -c conda-forge torchvision
conda install -c conda-forge matplotlib 


# snakemake  to download dataset 
# ref: https://snakemake.readthedocs.io/en/stable/getting_started/installation.html
$ conda install -c conda-forge -c bioconda snakemake
$ conda create -c conda-forge -c bioconda -n snakemake snakemake
$ conda activate snakemake

# To download the EPIC action segment labels, and build a new pickled dataframe for the train and test sets you can run
$ snakemake data/processed/{test_{,un}seen,train}_labels.pkl  --cores N # N the CPU number you want to cope with
$ snakemake -p gulp_all --cores N #This will download the RGB (220GB) and flow (100GB) frames, segment them using the test/train labels, and finally gulp the data. WARNING: The final few steps which involve gulping the frames will take a very long time if you aren't using a SSD - as such, we recommend that you locate the data directory on a SSD and symlink the directory here. 


----------HPC ------------------------------------------


----------Scalable ML------------------------------------------

'''>>> log in  the master node of ShARC. 
'''
# MAC oc Terminal Access ---ShARC   
##### To manipulate files and directories mac user better download a tool: Filezilla. 
# Unix Style "X11-terminal" access
# Open a Console Window
# In that window type 

'''
The University of Sheffield has two HPC systems:
SHARC Sheffield's newest system. It contains about 2000 CPU cores all of which are latest generation.
Iceberg Iceberg is Sheffield's old system. It contains 3440 CPU cores but many of them are very old and slow.
The two systems are broadly similar but have small differences in the way you load applications using the module system.
'''
    ssh -X acq18jg@iceberg.sheffield.ac.uk
or #(recommended) 
    ssh -X acq18jg@sharc.sheffield.ac.uk 
    cd /fastdata/acq18jg
# df -h dir_name # to check the available capacity!! 
# pwd: GAO-
      

# Windows :  via MobaXterm; MAC: filezilla;  remote host:  (port: 22) 
    sharc.sheffield.ac.uk   


'''>>> start an interactive session and Setup Jupyter Hub to run Notebook on HPC!!!!
'''
    qrshx -l gpu=1,rmem=40G (default 2G which may not enought for GPU )
    qrshx -l gpu=1,rmem=20G
        # https://docs.hpc.shef.ac.uk/en/latest/iceberg/GPUComputingIceberg.html
        qrshx -l gpu_arch=nvidia-m2070 -l rmem=7G  #(m2070: 8x Nvidia Tesla Fermi M2070 GPU)
        qrshx -l gpu_arch=nvidia-k40m -l rmem=13G  #(8x Nvidia Tesla Kepler K40M GPU)

or qrshx -P rse-com6012 -l rmem=60G # this command will get you a reserved node

# +++++++++++++++to check the availability of GUPs 
qhost -F gpu -h node100,node099,node126
# https://docs.hpc.shef.ac.uk/en/latest/sharc/groupnodes/big_mem_nodes.html

qrshx -l gpu=1,rmem=13G -P rse -q rse-interactive.q

qrshx -l rmem=60G

qrshx -l gpu=1,rmem=13G

module load apps/java/jdk1.8.0_102/binary 
module load apps/python/conda 
source activate jupyter-spark
module load libs/CUDA/10.2.89/binary
nvcc --version
nvidia-smi

# cd /fastdata/acq18jg/epic/Code/TRN-pytorch-master/
cd /data/acq18jg/Code/TRN-pytorch

# fro GPU 

# module load libs/CUDA/10.1.243/binary
# module load libs/CUDA/10.0.130/binary
# module load libs/CUDA/9.1.85/binary
# module load libs/CUDA/9.0.176/binary
# module load libs/CUDA/8.0.44/binary
# module load libs/CUDA/7.5.18/binary


qsub untitled1.sh 
qsub -P rse untitled0.sh 
qsub -P rse-com6012 Q2_HPC.sh



--------------2019--ML------------------------------------
# spyder

#%% (standard cell separator)

#Jupyter SHORTCUTS



"""
At the top of your notebook add this line.  xx.<TAB>,  xx.fun(<shift+tab>)
 %config IPCompleter.greedy= True 



注释
ctrl or cmd +/  comment multiple lines (only for jupyter . in spyder, it is cmd+ 1.  cmd4/5 comment block)
esc +B  below
esc +A  
D,D  
X
C  copy curent unit
V / shift+V    paste current copyboard to below 
ctrl+ enter  no move 
shift +enter  move
alt+ enter   insert a unit below and move the new one
Y  # change cell to code 
M  # change cell to markdown 

default argument . makes a difference when the default is a mutable object such
as a list, dict, or instances of most classes
"""

import sys
import pandas as pd
import math 
import numpy as np
import matplotlib.pyplot as plt
import getopt 
from IPython import display
import os
from functools import reduce


import json
import time
import re
from collections import Iterable, Counter
import pdb

%config InlineBackend.figure_format = 'retina'

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)


==================================
Notes @Shanghai Python #Jan 2023
==================================

Counter 排序
>>> from collections import Counter
>>> counter = Counter({'A': 10, 'C': 5, 'H': 7})
>>> counter.most_common()
[('A', 10), ('H', 7), ('C', 5)]
>>> sorted(counter.items())
[('A', 10), ('C', 5), ('H', 7)]



==================================
Notes @Diamond Python #Oct 2019
==================================


##  --------__getitem__() -----------

import copy
# Constants that can be used to index date of birth's Date-Month-Year
# example1 :
D = 0; M = 1; Y = -1

class Person(object):
    def __init__(self, name, age, dob):
        self.name = name
        self.age = age
        self.dob = dob
The [] syntax for getting item by key or index is just syntax sugar.

#When you evaluate a[i] Python calls a.__getitem__(i) (or type(a).__getitem__(a, i), 
#but this distinction is about inheritance models and is not important here). Even if the 
#class of a may not explicitly define this method, it is usually inherited from an ancestor class.
    def __getitem__(self, indx):
        print ("Calling __getitem__")
        p = copy.copy(self)

        self.age -= 1
        p.name = p.name.split(" ")[indx]
        p.dob = p.dob[indx] # or, p.dob = p.dob.__getitem__(indx)
        return p
    def desc(self):
        print('name:{}, age:{}, dob:{}'.format(self.name,self.age,self.dob))
p = Person(name = 'Jonab Gutu', age = 20, dob=(1, 12, 1999))
p.desc()
print(p[1].name) # print first (or last) name
p.desc()
p[0]
p.desc()
p.desc()
p[0]
p.desc()

# example2 :
class Building(object):
     def __init__(self, floors):
         self._floors = [None]*floors
     def __setitem__(self, floor_number, data):
          self._floors[floor_number] = data
     def __getitem__(self, floor_number):
          return self._floors[floor_number]

building1 = Building(4) # Construct a building with 4 floors

building1[2] = 'DEF Inc'
print( building1[2] )

##  --------------------------------

# Difference between __init__ and __call__
class Foo:
    def __init__(self, a, b, c):
        # ...
x = Foo(1, 2, 3) # __init__

class Foo:
    def __call__(self, a, b, c):
        # ...
x = Foo()
x(1, 2, 3) # __call__


# to say two object is same or not: 

print(np.allclose(data, data3)) # prints True



# letter frequency analysis 
from  collections import Counter
cnt = Counter('ciobsriittotpoaeoersretntonfgnyttltmnnpoueeomtjvs')
[(x, y) for x,y in cnt.items() if y==1 and x in 'encryption']
cnt 


cipher='ciobsriittotpoaeoersretntonfgnyttltmnnpoueeomtjvs'
def trans(cipher, n=5):
    li={}
    for i in range(math.ceil(len(cipher)/n)):
        li[i]=list(cipher[0+i*n: n+i*n if n+i*n < len(cipher) else len(cipher)])
    # print(li)
    df= pd.DataFrame.from_dict(li, orient='index').T
    df= df.fillna('')
    return df 
df=trans(cipher, 6)
print(pd.DataFrame(df,columns=[7, 6,0,3,5,2,4,1,8]))


# isinstance
 if (  isinstance(x, Iterable)) #  but be aware of the string is iterable too 

#arrow 
  u'\u2191' # up arrow  
  u'\u2193' # down arrow  

# show source code of a fun 
  funname ??

##  
 np.cumsum(items)  # calculated sum , the last one is same to reduce(lambda x,y:x+y, items) 
["{:.2%}".format(x) for x in np.cumsum( items) ]

## torch  tensor 

torch.dim()  # num of dimmensions
torch.shape # shape 
a.item() → #number only for one elements tensor. # a.tolist() for list. 
torch.mm(input, mat2, out=None) → Tensor
    Performs a matrix multiplication of the matrices input and mat2.

    a1= torch.randint(10, (3,2))
    a2= torch.randint(10, (2,3))
    print(a1, a2 )
    a1.mm(a2)
    torch.mm(a1, a2)

an underscore change the tensor in-place. That means that no new memory is being allocated by doing the operation, which in general increase performance, but can lead to problems and worse performance in PyTorch.

In [2]: a = torch.tensor([2, 4, 6])
tensor.add():
In [3]: b = a.add(10)
In [4]: a is b
Out[4]: False # b is a new tensor, new memory was allocated
tensor._add():
In [3]: b = a.add_(10)
In [4]: a is b
Out[4]: True # Same object, no new memory was allocated


torch.stack( ),  # add a new dimension
torch.cat(, dim= ) # don't add dims


# bracket 
[]  list 
()  tuple 
{}  dict  set 


#LISTlist   plus 

list + list = extend.
list+ ndarray = plus by elements
[-6, -14,-15]+ np.array([1,2,3])  # array([ -5, -12, -12])
[-6, -14,-15]+  [1,2,3] # [-6, -14, -15, 1, 2, 3]
# delete 
del a[2:4] 


#
#factorial 阶乘

math.factorial(3)

# repeat , replicate

x = np.array([[1,2],[3,4]])
np.repeat(x, 2)
np.repeat(x, 3, axis=1)
np.repeat(x, 3, axis=0)
np.repeat(x, [1, 2], axis=1)

#====function argument 
def add_end(L=[]):
    L.append('END')
    return L
add_end([1,2,3,4])

add_end(['a','b','c'])
add_end() # first calling is correct
add_end() # second calling get wrong. because default arg indicate a object, 
#and this object should  nbe  constant

Only the * (asterisk) is necessary. You could have also written *var and **vars. Writing *args and **kwargs is just a convention
*args is used to send a non-keyworded variable length argument list to the function
**kwargs allows you to pass keyworded variable length of arguments to a functio
def test_var_args( *argv, **kw):
    #print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)
    for k,v in kw.items():
        print("another {} through **kw:{}".format(k, v))

dic= {'a': 'a1','b': 123}

test_var_args('yasoob', 'python', 'eggs', 'test','ab', dic) 
test_var_args('yasoob', 'python', 'eggs', 'test','ab', **dic)

test_var_args('yasoob', 'python', 'eggs', 'test', 'ab'
              ,a='a1',b=123, c='c1'
              )


# map reduce filter 

from functools import reduce
items = [1, 2, 3, 4, 5]
list(map(lambda x: x**2, items)) #applies a function to all the items in an input_list
reduce(lambda x,y:x*y, items) #It applies a rolling computation to sequential pairs of values in a list
reduce(lambda x, y:x*10+y, items)
list(filter(lambda x: x > 2, items)) #creates a list of elements for which a function returns true.





# re slash backslash 
w=r'1\/2/b//dc'
chunk=re.split(r'(?<![\\\/])(?!//)/',w)
chunk



#rapidly create a matrix

x=np.array([1,3,4,5,6])
np.column_stack((np.ones(len(x)),x, x**2))
np.vstack([np.ones(5), x, x**2]).T
pd.DataFrame((x, x**2)).T


#lambda##
[i*i for i in range(0,10) if i>5]

#==========dict() dictionary
filter :
{k: v for k, v in points.items() if expression }


#merge 2 dicts 
dict1.update(dict2) #  update dict1 with dict2 
dict1.update((x, y*2) for x, y in dict1.items())
print dict1
#or 
dict(dict1, **dict2) # dict2 override dict1

# sort a dict 
sorted(dictname.items(), key=lambda x:x[1])

# 
term_postag_count.get('is',0)
get(key[, default])
'''Return the value for key if key is in the dictionary, else default. never raise keyerror'''

# a=set()
a.append() X , #append() is for list 
a.add()， a.remove() √
df.drop(columns=['B', 'C']) # drop column
df_d1 =pd.DataFrame(df_d, columns=['a','a1','c']) # select columns 
df.drop([0, 1]) # drop row by index 

#  regex   
re.split(r'[,.\s\n]+',line[:-1])

prog = re.compile(pattern)
result = prog.match(string)  # match from the beginning 
#search  from anywhere 

# ************************************************************
Note however that in MULTILINE mode match() only matches at the beginning of the string, whereas using search() with a regular expression beginning with '^' will match at the beginning of each line.

>>>
>>> re.match('X', 'A\nB\nX', re.MULTILINE)  # No match
>>> match= re.search('^X', 'A\nB\nX', re.MULTILINE)  # Match

match() and search() both return match object: 
Match objects always have a boolean value of True.

if match:
    process(match)


>>> m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
>>> m.group(0)       # The entire match
'Isaac Newton'
>>> m.group(1)       # The first parenthesized subgroup.
'Isaac'
>>> m.group(2)       # The second parenthesized subgroup.
'Newton'
>>> m.group(1, 2)    # Multiple arguments give us a tuple.
('Isaac', 'Newton')

m = re.match(r"(\w+) [a-z]{2} (\w+)", "Isaac xx Newton, physicist")
m.groups() #('Isaac', 'Newton')  only return groups.
m.group(0) # 'Isaac xx Newton'
# ************************************************************
# set() 
s1={1,2,3,4,'a','b','c'}
s20={3,4,'b','c'}
s2={3,4,'b','c','d'}

s2.issubset(s1)
s2.union(s1,s2) #union
s2.difference(s1)  # subtract
s2.intersection(s1)  # join 
s2.symmetric_difference(s1) # xor
s2.update(s1) #   doesn'r return new set ,but return updated s2 


# numpy concatenate 
a= np.array([[1,2], [3,4]])
b=np.array([5,6])
c=np.array([[0.1,0.2],[0.3,0.4]])

#same shape    rbind cbind
np.concatenate((a,c), axis=1)#√
np.hstack((a,c))#√
np.concatenate((a,c), axis=0)#√
np.vstack((a,c)) #√

a= np.array([[1,2], [3,4]])
b = np.array([[4, 5], [6, 7]])
print(a.flatten())
print(b.flatten())
c = np.c_[a.flatten(), b.flatten()]

# different shape 
b1=np.array([[5],[6]])
np.concatenate((a,b1), axis=1)#√ 
np.hstack((a,b1))#√ 
np.concatenate((a,b), axis=0) #x
np.vstack((a,b)) #√ 
np.vstack((a,b1.T)) #√ 

np.concatenate((a,b), axis=None)#√ 
np.concatenate((a,b1), axis=None)#√ 
np.concatenate((a,c), axis=None)#√ 

# dataframe add a columns /append/ concat/ stack 
cnt= pd.Series(xx, name='mv_cnt')  # xx iterable 
pd.concat([df_tagname, cnt], axis=1)
df['newcolumn'] = newcolumn 
# dataframe add/append  rows
p= p_group.loc['a','b','c']
total= p.sum()
total.name='total'
p.append(total)

#### dataframe join , merge 
## Use merge, which is inner join by default:

pd.merge(df1, df2, left_index=True, right_index=True)
Or join, which is left join by default:

df1.join(df2)
Or concat, which is outer join by default:

pd.concat([df1, df2], axis=1)
Samples:

df1 = pd.DataFrame({'a':range(6),
                    'b':[5,3,6,9,2,4]}, index=list('abcdef'))

print (df1)
   a  b
a  0  5
b  1  3
c  2  6
d  3  9
e  4  2
f  5  4

df2 = pd.DataFrame({'c':range(4),
                    'd':[10,20,30, 40]}, index=list('abhi'))

print (df2)
   c   d
a  0  10
b  1  20
h  2  30
i  3  40
#default inner join
df3 = pd.merge(df1, df2, left_index=True, right_index=True)
print (df3)
   a  b  c   d
a  0  5  0  10
b  1  3  1  20

#default left join
df4 = df1.join(df2)
print (df4)
   a  b    c     d
a  0  5  0.0  10.0
b  1  3  1.0  20.0
c  2  6  NaN   NaN
d  3  9  NaN   NaN
e  4  2  NaN   NaN
f  5  4  NaN   NaN

#default outer join
df5 = pd.concat([df1, df2], axis=1)
print (df5)
     a    b    c     d
a  0.0  5.0  0.0  10.0
b  1.0  3.0  1.0  20.0
c  2.0  6.0  NaN   NaN
d  3.0  9.0  NaN   NaN
e  4.0  2.0  NaN   NaN
f  5.0  4.0  NaN   NaN
h  NaN  NaN  2.0  30.0
i  NaN  NaN  3.0  40.0


# rep * , concatenate + 
states=['NY']*5+['FL']*4+[0]*2
# get a result ['NY', 'NY', 'NY', 'NY', 'NY', 'FL', 'FL', 'FL', 'FL', 0, 0]



#
np.linspace(start , stop , num )
#Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`].





## 
def get_catdog_index(ds, label_set={3,5}):
    return np.where( [x[1] in label_set for x in ds])[0]

x=list(range(10))
np.random.shuffle(x)
np.argmax(x) # return the index of the max element.
np.argmin(x)
np.argsort(fimp) # return sorted(asc) index 

""" #if expression  ifelse Rprograme
x= a if (True/expression) else b  # while in R : x <- if( expression) a else b,  or ifelse(expression, yes, no)
"""

immutable types:  numbers, string, tuples.


"""


"""
import os style instead of from os import *. 
This will keep os.open() from shadowing the built-in open() function 
which operates much differently.
 """   


"""

filter return a iterator object, and ```only could be used once```. once exhauted, it will be []
"""print('a is:','b')


iterator: ```and only could be used once. once used, it will be []```
"A 【container】 object (such as a list) produces a fresh new iterator each time you pass it to the iter() function or use it in a for loop. Attempting this with an iterator will just return the same exhausted iterator object used in the previous iteration pass, making it appear like an empty container." 



testloader= (1001, 1002, 1003, 1004)
itr= iter(testloader) 


print(list(itr))  # list here  will exhaust itr firstly, then in the next for loop, it will return nothing.  
for i in range(5):
    try:
        print('-'*3, 'echo:',i)
        x=next(itr)
        print(x)
    
    except StopIteration:
        print('error',i,' end of iteration')
# print(list(itr))  # while list is behind for loop, then here it will print only []





hash-based structure: 
# SET  and DICT give their element/keys in NO particular order.
# set_1 = set([5, 2, 7, 2, 1, 88])
# #sorted(set_1, key= lambda x:x*-1, reverse=1)

# sequence-based structure
# string, list, tuple, file-stream  give their element in ORDER. 

"x in dictname"   is  nicer than "dictname.has_key(x )"

getopt() # https://pymotw.com/2/getopt/

"""  

The getopt function takes three arguments:

The first argument is the sequence of arguments to be parsed. 
This usually comes from sys.argv[1:] (ignoring the program name in sys.arg[0]).
注意参数一定要按顺序，
例如 getopt.getopt(['-a', '-ba', '-ca', 'val1','-val2'], 'ab:c:v:')
opt以第一个非opt的参数出现为止，最后一个“-val2"会被视为普通参数 
Option processing stops as soon as the first non-option argument is encountered.


pt, arg = getopt.getopt(['-abrg1', '-carg1', 'val1','-val2'], '1abc:v:rg')
短参数可以写在一起 [‘-abrg1’]， 也可以分开写。['-a',gett'-b','-r','-g','-1'], 
但是前边出现的参数， 一定要在second argument (eg: '1abc:v:rg') 中出现， 否则error

returns a sequence of (option, argument) pairs and a sequence of non-option arguments.

The second argument is the option definition string for single character options. If one of the options requires an argument, its letter is followed by a colon.
The third argument, if used, should be a sequence of the long-style option names. Long style options can be more than a single character, such as --input or --output_file. The option names in the sequence should not include the -- prefix. If any long option requires an argument, its name should have a suffix of =.


$ python getopt_example.py -o foo
$ python getopt_example.py -ofoo
$ python getopt_example.py --output foo
$ python getopt_example.py --output=foo

"""

\s white space
  \d  \w  [A-Za-z0-9]
\b  match at boundry of words
\S  \D  \W   negate the meaning of the lowcase .

# version info 
print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)


# directory
print(os.getcwd()) 
print(os.pardir)
print(os.curdir)
print(os.path.expanduser('~'))  # python doesn't recognise tilde , have to do ti manually

/Users/gaojiejun/wd/python
..
.
/Users/gaojiejun


"""

# save pd dataframe 
# input output 
df.to_csv('births1880.csv',index=True,header=True)
df.to_csv('my_first_csv.csv')

path=os.getcwd()
path
df.to_csv('demodf5.csv')
df.to_csv( r'./datafilename/demodf4.csv') # df.to_csv(os.getcwd()+r'/datafilename/demodf4.csv')

''' datafilename should be existing already. otherwise error raised''' """
df=read_csv( r'./demodf4.csv', index = False)# df=read_csv(os.getcwd()+r'/demodf4.csv')
df.to_excel('Lesson10.xlsx', sheet_name = 'testing', index = False)
# os.curdir is a string representing the current directory (always '.')
#  os.pardir is a string representing the parent directory (always '..')
os.chdir(r"../datafiles")  # change pwd to specified dir


import zipfile
zip1 = zipfile.ZipFile('./master.zip', 'r')
for name in zip1.namelist():
    zip1.extract(name, '.')
    
# output  #with open()
"""
'r' when the file will only be read, 
'w' for only writing (an existing file with the same name will be erased), 
'w+' Opens a file for both writing and reading. Overwrites the existing file if the file    exists. If the file does not exist, creates a new file for reading and writing.
'a' opens the file for appending; any data written to the file is automatically added to the end. 
'a+' Opens a file for both appending and reading. The file pointer is at the end of the file if the file exists. The file opens in the append mode. If the file does not exist, it creates a new file for reading and writing.
'r+' opens the file for both reading and writing. The mode argument is optional; 
'r' will be assumed if it’s omitted.
"""


''' terms is a dict '''
f= open("./_textprocessing/WriteHere/lab2.txt","a")
    #f=open("guru99.txt","a+")
f.write("\nf======={}========\n".format(time.strftime('%X %d-%b-%Y %Z')))
for i in terms.items():
    f.write(str(i))
f.close()

'''as same as :'''
with open("./_TextProcessing/WriteHere/lab2.txt","a") as text_file:
    text_file.write("\nwith======={}========\n".format(time.strftime('%X %d-%b-%Y %Z')))
    text_file.write(json.dumps(terms))  
    
with open("copy.txt", "w") as file:
    file.write("Your text goes herejjg")


with open('myfile.txt') as f: 
"""
The with statement allows objects like files to be used in a way 
that ensures they are always cleaned up promptly and correctly.


with open("/data/acq18jg/ass2/q1_result.txt","a+") as f:
    
    f.write("\n@time:======={}========\n".format(time.strftime('%X %d-%b-%Y %Z')))
    f.write(json.dumps(a)) 

import re 
x=[]
with open("q1_result.txt","r") as f:
    tm=re.compile(r'^@time')
    for line in f: 
        if line.strip() and not tm.match(line):
            print(line)
            x.append(line.replace("\n", ""))

x

# ls=[np.random.randint(low=0,high=5) for i in range(1000)]
## np.random.randint(high=5) #default 0- high
# cnt=0
# for i in set(ls):
#     print(i,': ', ls.count(i))
#     cnt+= ls.count(i)
# print('totsl: ', cnt)


YearMonth = ALL.groupby([lambda x: x.year, lambda x: x.month])
df.State.apply(lambda x: x.upper())




# timer:

t_start= time.time()
...
t_end = time.time()
t_diff = t_end - t_start

#time format:    https://docs.python.org/3/library/time.html?highlight=time#time.struct_time
time.strftime('%X %x %d-%b-%Y %Z') #'17:58:44 30-Oct-2019 GMT'
time.strftime("%Y-%m-%d %H:%M:%S")

# datetime / time 
t= x['stop_timestamp'] # 00:00:01.89
pd.to_timedelta(t) #  00:00:09.49 
pd.to_datetime(t) # 2020-04-18 00:00:03.370




# zip()
a = list(['1','2','3'])
b = ("Jenny", "Christy", "Monica", "Vicky")
x = zip(a, b)
print(list(x))

#lambda 

x=list(range(5))
list(map(lambda x: x+2001, x))
x=list(range(2001,2006))



# print() print
year = 2016
event = 'Referendum'
f' Results of the {year} {event}'


yes_votes= 42_572_654
no_votes= 43_132_495
percentage= yes_votes/ (yes_votes+ no_votes)
'{:^19.2f} Yes votes {:2.2}'.format(yes_votes, percentage)


formating 
# =============================================================================
# 'd'	Signed integer decimal.	
# 'i'	Signed integer decimal.	
# 'o'	Signed octal value.	-1
# 'u'	Obsolete type – it is identical to 'd'.	-8
# 'x'	Signed hexadecimal (lowercase).	-2
# 'X'	Signed hexadecimal (uppercase).	-2
# 'e'	Floating point exponential format (lowercase).	-3
# 'E'	Floating point exponential format (uppercase).	-3
# 'f'	Floating point decimal format.	-3
# 'F'	Floating point decimal format.	-3
# 'g'	Floating point format. Uses lowercase exponential format if exponent is less than -4 or not less than precision, decimal format otherwise.	-4
# 'G'	Floating point format. Uses uppercase exponential format if exponent is less than -4 or not less than precision, decimal format otherwise.	-4
# 'c'	Single byte (accepts integer or single byte objects).	
# 'b'	Bytes (any object that follows the buffer protocol or has __bytes__()).	-5
# 's'	's' is an alias for 'b' and should only be used for Python2/3 code bases.	-6
# 'a'	Bytes (converts any Python object using repr(obj).encode('ascii','backslashreplace)).	-5
# 'r'	'r' is an alias for 'a' and should only be used for Python2/3 code bases.	-7
# '%'	No argument is converted, results in a '%' character in the result.	
# =============================================================================

'%s %s'% ('one','two')  #old  print
'{}{}'.format ('one','two') # new print
'%-10s' % ('test',)
'{:10}'.format('test') 
# default left alain while the old style default right
 #aligned
'{:#^11}'.format('test') #< left(default) , ^middle, >right

print('{:#<11}'.format('test') )
print('{:#<11} i donnot know who i am{:+1006.1f}old'.format('test',+23.2) )
print('{:11} helloworld{:+5.1f}old'.format('test',+23.2) )

data = {'first': 'Hodor', 'last': 'Hodor!'}
print('{first} {last}'.format(**data) )
print( 'first last'% data)
# =============================================================================
# Number        Format  Output      Description
# 3.1415926     {:.2f}  3.14        2 decimal places
# 3.1415926     {:+.2f} +3.14       2 decimal places with sign
# -1            {:+.2f} -1.00       2 decimal places with sign
# 2.71828       {:.0f}  3           No decimal places
# 5             {:0>2d} 05          Pad number with zeros (left padding, width 2)
# 5             {:x<4d} 5xxx        Pad number with x’s (right padding, width 4)
# 10            {:x<4d} 10xx        Pad number with x’s (right padding, width 4)
# 1000000       {:,}    1,000,000   Number format with comma separator
# 0.25          {:.2%}  25.00%      Format percentage
# 1000000000    {:.2e}  1.00e+09    Exponent notation
# 13            {:10d}          13  Right aligned (default, width 10)
# 13            {:<10d} 13          Left aligned (width 10)
# 13            {:^10d}     13      Center aligned (width 10)
# =============================================================================
list([data['first'],data['last']])

# as a tuple list 
t=list([data['first'],data['last']])
print('{0} is：{1}'.format(t[0],t[1]) ) 

# as a object 
class Plant(object):
    type = 'tree'
    kinds = [{'name': 123}, {'name': 456.6}]

'{p.type}: {p.kinds[0][name]}'.format(p=Plant())
p=Plant() 
'{}: {:12.2f}'.format(p.type, p.kinds[0]['name']) 


'{:%Y-%m-%d %H:%M:%S::%f}'.format(datetime(2001, 2, 3, 4, 5,5,999))
#%X %d-%b-%Y %Z  03:41:58 05-May-2020 BST
# =============================================================================
# %a	Weekday, short version	Wed	
# %A	Weekday, full version	Wednesday	
# %w	Weekday as a number 0-6, 0 is Sunday	3	
# %d	Day of month 01-31	31	
# %b	Month name, short version	Dec	
# %B	Month name, full version	December	
# %m	Month as a number 01-12	12	
# %y	Year, short version, without century	18	
# %Y	Year, full version	2018	
# %H	Hour 00-23	17	
# %I	Hour 00-12	05	
# %p	AM/PM	PM	
# %M	Minute 00-59	41	
# %S	Second 00-59	08	
# %f	Microsecond 000000-999999	548513	
# %z	UTC offset	+0100	
# %Z	Timezone	CST	
# %j	Day number of year 001-366	365	
# %U	Week number of year, Sunday as the first day of week, 00-53	52	
# %W	Week number of year, Monday as the first day of week, 00-53	52
# =============================================================================


# matrix  
a = np.ones((2,3), dtype=int) / np.zeros, empty, /np.random.random((2,3))
b = np.random.random((2,3))

np.random.randomj # uniformly (0,1)
np.random.rand                 Uniformly distributed values.
np.random.randn                Normally distributed values.
np.random.ranf                 Uniformly distributed floating point numbers.
np.random.randint              Uniformly distributed integers in a given range.
np.random.shuffle(data)
# or 
c= np.array(list(range(15))).reshape(3,5)
for i in c:
    print('{:^5.2f}{:5}{:5}'.format(i[0], i[1],i[2]))

    
# DataFrame =====DATAFRAME========================================================================  

### summarize a dataframe 
def summary(dataset, name='total'):
    ds= dataset.copy() #  because we are gonna add columns , so it's better to use a copy 
    ds['duration']= pd.to_timedelta(ds['stop_timestamp'])- pd.to_timedelta(ds['start_timestamp'])
    ds['frames']= ds['stop_frame']-ds['start_frame']+1

    df= pd.DataFrame()
    df['uid']= pd.Series(ds['participant_id'].count())
    # count
    for c in ['video_id']:
        if c not in ds.columns: continue
        temp= pd.Series(ds[[c]].groupby(c).size().count())
        df[c]= temp
    # distinct 
    for c in ['verb','verb_class','noun','noun_class','narration']:
        if c not in ds.columns: continue
        temp= pd.Series(ds[[c]].groupby(c).size().count())
        #print(type(temp))
        df[c]= temp
    #sum 
    for c in ['duration','frames']:
        if c not in ds.columns: continue
        temp= pd.Series(ds[c].sum()) # sum of a pd.Series. ds[[c]].sum() = sum of a df
        #print(c,type(temp))
        df[c]= temp
    df.index= [name]
    return df 




sorteddf=df.sort_values(['age', 'grade'], ascending=[True, False])
df.iloc[::-1] # inverse sort simply
#### warning:  after sort_value, sorteddf['age'][0] may get wrong anwer, you should use sorteddf['age'].iloc[0] to get the smallest age.


#to check the structure of the df
df.dtypes  


#To delete a column, or multiple columns, use the name of the column(s), and specify the “axis”
data = data.drop("Area", axis=1)
#Rows can also be removed using the “drop” function, by specifying axis=0. Drop() removes rows based on “labels”, rather than numeric indexing

# sample with/out replacement 

def split_dataset(dataframe, split_rate):
    df_size= len(dataframe)
    sample = np.random.choice(np.arange(df_size),size= df_size, replace=False) # same as : np.random.shuffle( ) 
    train_split= np.arange(df_size)<= df_size*split_rate -1 # select train set 

    train_set= dataframe.iloc[sample[train_split] ]
    test_set= dataframe.iloc[sample[~train_split] ]
    print('total {}, train set {} ,test set {}'.format(df_size, len(train_set), len(test_set)))
    return (train_set, test_set)

train_set,test_set =split_dataset(air_quality, 0.7)


def random_batch(ds, batchsize, is_group=False):
    '''randomly distribute ds into batches with size= batchsize, if is_group= True, then batchsize= group_num'''
    ds=len(ds)
    batch_num= math.ceil(ds/batchsize) 
    batches_idx=[]

    rnd= np.random.choice(np.array(range(ds)),size= ds,replace= False)
    rnd= np.append(rnd, np.repeat(np.nan,batch_num*batchsize- ds )) # make the elements number exactly batch_num*batchsize
    m=rnd.reshape((batch_num, batchsize), order='C') # matrix  byrow ,  order='F' by columns
    if(is_group): m= m.T 
    for i in m:
        batches_idx.append(i[ ~np.isnan(i)].astype(int)) # transfer dtype . 
        #print(i[ ~np.isnan(i)].astype(int))
    print('random batched into {} groups'.format(len(m)))
    return batches_idx
def k_fold( ds, K):
    ''' return a list of (training, validation) index'''
    x= np.arange(len(ds))
    idx1=random_batch(ds, batchsize=K, is_group=True )  
    rls=[]
    for j in idx1:
        valid_idx = j
        noselected =[k not in j for k in x]
        train_idx= x[noselected]
        np.random.shuffle(train_idx)
        rls.append((train_idx, valid_idx))
    return np.array(rls) 


#avoid to create a DF from another existing DF(which has original columnnames) , if have to , don't set the index or columns, because it will become filtering from the original DF. 
# you could set them in independent lines.
data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df=pd.DataFrame(data2)
df
np.array(df)
pd.DataFrame(np.array(df))
pd.DataFrame(data2, index=['first', 'second'])
pd.DataFrame(df, index=['first', 'second'])

f1=pd.DataFrame(data2) /
df1.columns= df.columns+'_1'
df1.index= ['fisrt','second']
df1

#slicing & dicing $ selecting 
#DataFrame The basics of indexing are as follows:

Operation	Syntax									Result
------------------------------------------------------------------------
Select 		column			   		df[col]			Series
Select 		row by label			df.loc[label]	Series
Select 	    row by integer_location	df.iloc[loc]	Series
Slice 		rows					df[5:10]		DataFrame
Select 		rows by boolean vector	df[bool_vec]	DataFrame
------------------------------------------------------------------------

# DataFrame's  loc[] can new new element directly whilst iloc[] cannot.
# loc[] gets rows (or columns) with particular labels from the index.
# iloc[] gets rows (or columns) at particular positions in the index (so it only takes integers).
newseries = df.loc[ (df['BBB'] > 25) & (df['CCC'] >= -40), :'CCC'] #  loc accept boolean
newseries
newseries = df.iloc[ (df['BBB'] > 25).values & (df['CCC'] >= -40).values, :3]  # iloc no boolean
newseries




#filter a list by a boolean list.  list don't support boolean selecting unless cast it to np.array. 
# selecting the non-digital columns: 
np.array(df.columns)[np.array([not isinstance(x, numbers.Number)  for x in df.loc[0]])]

# indexing , selecting # 
# List can't take boolean index. so have to use np.array or pd.Series
l=list(range(1950, 2011))
np.array(l)[np.array(l)>2000]
pd.Series(l)[pd.Series(l)>2000]





Y=list(range(5,10))
prd=list(['a','b','c','d','r'])
p= pd.DataFrame(prd, index= Y, columns= ['Prediction'])
p.loc[1,'lab']='new'
p.iloc[1,2]='__newlab'


#  Series * +
a=['aa','ba','ca']
b=[1,2,3]
a*2
a+'_1'  # wrong

[a+'_1' for a in a] #√
list(pd.Series(a)+ '_1') # √
pd.Series(b)+1 # √
pd.Series(a)*2 # √
pd.Series(b)**2 # √


======matplotlib.pyplot   plt ===================


plt.barh(y_pos, height)  # horizontal bar 
# add background grid 
plt.rc('axes', axisbelow=True)
plt.grid(b=None, which='major', axis='both', **kwargs)

#plt.figure(dpi=1200)

x=list(range(1,11))
y= [0.484, 0.523,0.547,0.551,0.541,0.539,0.529,0.528,0.520,0.519]
plt.plot(x, y, 'x-r') # '[marker][line][color]'
plt.xticks(x)
plt.ylim(0.45  , 0.6)  
plt.xlabel('rank K')
plt.ylabel('precision ')
plt.title("average precision_at_rank (top 10)")
# plt.legend(('$w$', '$z$'))#plt.legend()

plt.tight_layout() # to make sure label would not be cut off 

plt.savefig("/home/acq18jg/Figure/Lab4_fig1.pdf", format='pdf')
plt.savefig(os.path.expanduser('../Figure/testpyspark.pdf'), format='pdf')

plt.savefig(os.path.expanduser('~/sheffield/precision_at_rank.eps'), format='eps') # extremely high quality




def training_plot(df, cols=['train_loss','val_loss'], markers=None, colors= None,
                  main='Training Monitoring',xlabel='Epochs', ylabel='loss', save_as_pdf=None ):
    '''plot the monitoring loss matrix. '''
    x= df.index
    
    fmt= ['-']*len(cols)
    if markers: fmt= list(map(lambda x: x[0]+x[1], zip(fmt,markers)))
    if colors:  fmt= list(map(lambda x: x[0]+x[1], zip(colors, fmt)) )
    for i,col in enumerate(cols):
        plt.plot(x,df[col], fmt[i], label=col)
        
    plt.legend(loc = 'best')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.xticks(x[np.arange(0, len(x),5)])
#     plt.ylim([0, 1.2])
    plt.title(main)
    
    plt.tight_layout()
    if save_as_pdf: plt.savefig(save_as_pdf, format='pdf')
    plt.show()





# colors=['salmon','lightseagreen']


def bar_graph( ds, index_col= None, bar_col= None,
              plotname='verb_class', colors=['salmon'], xlabel='', ylabel='', 
              save_as_pdf=''):
    ''' ds is a single column of a pd.dataframe. or a Series.'''
    
    index= np.arange( len(ds))
    my_colrs= colors
    if isinstance(ds, pd.Series) or len(ds.columns)==1 : 
        x_ticks= ds.index
        bar_col= ds 
    else: 
        x_ticks = ds[index_col]
        bar_col= ds[bar_col]
        
    if len(colors)==1 : my_colrs = colors* (len(ds))
    
    
    plt.bar(index, bar_col ,#width=1/2, 
            color= my_colrs)

    plt.yscale('log')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15) 
    plt.xticks(index, x_ticks, fontsize=10, rotation=90,ha="center")
#     plt.ylim(0 , 0.3)#  
    plt.title(plotname, fontsize=18)
    plt.tight_layout()
    if save_as_pdf: plt.savefig(save_as_pdf, format='pdf')
    plt.show()
    plt.close()

plt.rcParams["figure.figsize"] = (20,6) # the canvas inches
plt.margins(x=0.01)# to contron the margin sapce inside the border.
ds= pd.DataFrame(verb_class).join(verb_dict)
ds.columns =['cnt']+ list(ds.columns[1:])
ds= ds.sort_values(['cnt'], ascending=[False])
bar_graph(ds, index_col='class_key', bar_col='cnt', ylabel='Numer of actions',
          colors=['salmon']*8+ ['lightskyblue'] *(len(ds)-8),
         save_as_pdf='../../deliverable/Figure/verb_class.pdf')

#  
evaluation['auc']= pd.DataFrame(roc).iloc[:,2]
my_colrs = ['tomato']+['lightseagreen']* (len(evaluation['auc'])-1)
# plot roc AUC
marks=['.','o','*','+','x','H','1','2']
for i in range(len(roc)):
    fpr, tpr, auc, threshold= roc.iloc[i]
    plt.plot(fpr, tpr, 
             marker= marks[i],
             alpha=0.7,
#             color= color[i],
             label= 'PCA= {:4d} , AUC= {:0.2f}'.format(evaluation['No_pca'][i], evaluation['auc'][i]))
#     plt.plot(fpr, tpr, 'b', label = 'PCA={}, AUC = {:0.2f}'.format(4,roc_auc))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc = 'best')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic of Gaussian NB')
plt.show()


# plot a histgram 
'''
when plot a histgram using statistical data ( two fields like: claim_amount, cnt ), 
take a pre-processing :
    ds= np.repeat(df.claim_amount, df.cnt) 
'''

ds_1= ds#[ds>80]
name='distribution of points claimed'
ticks= list(range(0,max(ds_1),5000))
plt.hist(ds_1,bins = 200,log=True)
plt.title(name)
# plt.xlim([0,3000])
plt.xticks( ticks #, labels= [ str(i/10000)+'w' for i in ticks] 
           ,fontsize=8
          )
plt.xlabel('points claimed')
plt.ylabel('frequency')
plt.show()
# plt.savefig(name+'.pdf', format='pdf') 
plt.savefig(os.path.join(savedir,'Q2_1_claim_amt.pdf'), format='pdf')
plt.close()

# grouped bar chart 

def bar_group(ds,  bars_cols,index_col=None, xlabel='',ylabel='', 
    plotname='Fig.Name',colors=0, alpha=1 , xtick_rotation=0, save_as_pdf=''):
    '''
    plot a grouped bar chart
    '''
    N= len(ds) 
    grps= len(bars_cols)
    index = ds[index_col] if index_col else ds.index 
    barWidth = 1/(grps+1)   # if group of n, then the barWidth = 1/(n+1)
    for i, col in enumerate(bars_cols):
        r= [x + barWidth*i for x in range(N)]
        bars= ds[col]
        if colors:
            plt.bar(r, bars,width=barWidth, edgecolor='white', label=col, color=colors[i], alpha= alpha)
        else:
            plt.bar(r, bars,width=barWidth, edgecolor='white', label=col, alpha= alpha) 
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.yticks(fontsize=8)
    plt.xticks(ticks= [r + (0.5-barWidth) for r in range(N)], labels=index,fontsize=8, rotation=xtick_rotation,ha="right")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(plotname)
    plt.tight_layout() 
    if save_as_pdf:
        plt.savefig(save_as_pdf, format='pdf')
    plt.show()



# error bar 
labels = ds4.index
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, (ax0) = plt.subplots()
ax0.bar(x - width/2, height= ds4['mae_mean'] , yerr=ds4['mae_std'], width= width, label='mae', #align='edge',
        alpha=0.7, color='tomato', 
        capsize=10)
ax0.bar(x + width/2, height= ds4['rmse_mean'], yerr=ds4['rmse_std'],  width= width, label='rmse', #align='edge',
        alpha=0.7, #ecolor='black', 
        capsize=10 )
ax0.set_xlabel('als versions')
ax0.set_ylabel('mean and stddev(errorbars)')
ax0.set_title('Evaluation of 3 ALS versions \n')
ax0.set_xticks(x)
ax0.set_xticklabels(labels)
ax0.legend() #plt.legend(bbox_to_anchor=(1, 1)) # to locate legend outside top right
plt.tight_layout()


# set grid  ggplot similar

plt.yscale('log')
plt.rc('axes', axisbelow=True)# grid below the charts
ax0.set_facecolor("#E8E8E8")
#removing top and right borders
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
plt.grid(b=True, which='major', axis='both', color='white') #'linen'
plt.tight_layout()



# -----subplot--------------------------------------------------------------------  

def sub_training_plot(plt, df, cols, index_col= None,
                      main='',xlabel='xname', ylabel='yname', ylims=None, xlims=None, iflegend= 1 ):
    x= df[index_col] if index_col else df.index 
    for col in cols:
        plt.plot(x, df[col],'-', label=col) #plt.bar(x, df[col],width=2/4, label='fps' )
    if ylims and len(ylims)== 2: # a tuple 
        plt.set_ylim(ylims)
    if xlims and len(xlims)== 2: # a tuple 
        plt.set_xlim(xlims)
    if iflegend: plt.legend(loc = 'best')
    plt.set_title(main, fontsize=11)


fig, axs = plt.subplots(1, 3, figsize=(9,3)) # figsize inches.
# fig.suptitle('\n plot NAME', fontsize=14)
## axs[0, 0].plot(x, y)
# axs[-1, -1].axis('off')
for i,(name,v) in enumerate(dict_model.items()): 
    series= v
    sub_training_plot(plt= axs.flatten()[i], df=series, cols=['train_loss','val_loss'], 
                      main= name, ylims= [.9, 2.], xlims=[-1,41] )

for ax in axs.flat:
    ax.set(xlabel='epoch', ylabel='loss')
    
# Hide x labels and tick labels for top plots and y ticks for right plots. // at last 
for ax in axs.flat:
    ax.label_outer()

fig.tight_layout()
fig.savefig('../../deliverable/Figure/training_loss.pdf', format='pdf')




subplot 2021-06 


rsl={}
metrix= np.matrix([
         ['dy_active_rate','both_active_rate'],
        ['dy_duration','both_duration'],
         ['dy_tax_revenue_acc','both_amt']])
# ['dy_active_rate', 'dy_duration','dy_tax_revenue_acc',
#                            'both_active_rate','both_duration', 'both_amt']

fig, ax = plt.subplots(metrix.shape[0],metrix.shape[1],  sharey='row',figsize=(8,9) 
                      )
name= 'group11_dylite_&_hgame_virtualAB'

for k,metric in enumerate(metrix.flat):
    ht={} #  hypothesis test datum.
    # metric='dy_tax_revenue_acc'
    
    for i in ds.tag.unique():
        x= ds[ds.tag==i]

        ht['date']=x.date.tolist()
        ht[i]=x[metric].tolist()
        ax.flat[k].plot(x.date,x[metric],label=i)

    ht_df= pd.DataFrame.from_dict(ht)
    ax.flat[k].axvline(x= 30, linestyle='--',color= 'r', linewidth=1  ) 
    if k==0: ax.flat[k].legend(loc = 'best')
    ax.flat[k].set_title(metric)
    tck= ht_df.index%7==0
    ax.flat[k].set_xticks(ht_df.index[tck], minor=False)
    ax.flat[k].set_xticklabels(ht_df.date[tck], rotation=45, fontdict=None, minor=False)


    ### save the hypothesis test results. 
    v= ht_df.loc[:29,]
    # stats.ttest_rel(v['control'],v['group_test'])
    p, t_sta, t_pvalue= two_sample_test(v['group_test'],v['control'],0)
    v= ht_df.loc[30:,]
    obs_p, obs_t_sta, obs_t_pvalue= two_sample_test(v['group_test'],v['control'],0)

    rsl[metric]={'before_var_p':p, 'before_t_sta':t_sta, 'before_t_pvalue':t_pvalue,
             'obs_var_p':obs_p, 'obs_t_sta':obs_t_sta, 'obs_t_pvalue':obs_t_pvalue }


    
for i,axk in enumerate(ax.flat):
    axk.set(xlabel='date', ylabel=('','','duration/s','','amout/分','')[i])
#     axk.label_outer()   
fig.suptitle(name, fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) #tight_layout() only considers ticklabels, axis labels, and titles. Thus, other artists may be clipped and also may overlap.

fig.savefig(name+'.pdf', format='pdf')


rsl_df= pd.DataFrame.from_dict(rsl).T
rsl_df




==================================
notes @R 
==================================

reverse :  rev(vector).  
vector[-1] #will get elements excluding 1st element 

a <- c('ab',123,'c')
b <- list('ab',123,'c')
# a’s result will turn 123 to '123' ,because "concattenate" will make all the elements the same datatype.
# if you want to do this , only list. 
# and if you want get the number of it ,  then unlist() will get a vector 


# & and && , | and || 
&, | do operation element by element

&&, || only do the first elemet 
when it is a vector, then use & 

x <- 1:3
y <- 4:6

x <= 1 
y < 5
x <= 1 & y < 5
x <= 1 && y < 5


x < 1
y > 5
x < 1 | y > 5
x < 1 || y > 5
```

Pr_defective<- function(arg1=, arg2=, arg3=, ,,){  statement }
  # X <- function(...) Pr_defective(...)
  # y <- function() X(N1 = 10000, N2 = 20000,p1 = 0.05, p2 = 0.02, n=200)
  

==================================
notes @ terminal  croppdf 
==================================

pdfcrop input.pdf output.pdf # especially when you have to insert a excel picture(table)(saved as pdf) into latex files. it's useful .
