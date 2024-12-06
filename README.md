# About
This repository contains experiments done on the data set from Wibisono et. al. [Science Direct](https://www.sciencedirect.com/science/article/pii/S2352340920308180)
using the library [STLCG](https://github.com/StanfordASL/stlcg).

## Usage

If you want to reproduce the exact results, you can use the provided Dockerfile, or run the Image under https://hub.docker.com/r/pwittlinger/declarestlexperiments
Be sure to save the files which are being output by mounting a local folder to the container. It contains all necessary dependencies and the relevant data.

```
docker -run -v <path/to/local>:/outputs/ declarestlexperiments
```

If you would instead like to try some other constraints you can Edit the file _Constraints.txt_. 
However, you will also need to edit the Code in _stlExperiments.py_ to match the corresponding input to the Constraints (cf. line 190)


## Declare2STL Parser

If you want to use the implemented parser (_declare2stl_parser.py_) or check some constraints individually, the following provides an overview. For further details on the stlcg library, see also https://www.youtube.com/watch?v=hrjt6abUPDA

```
my_parser = Parser()

########## Simple Print Example ##########

some_test_strings = [ #see syntax in the paper
    "Existence[0,2](a[t]>1)",
    "Response[0,m][5,8](a[t]>2, dis(d[t]))"
]

for t in some_test_strings:
  print(f"Text: {my_parser.transform_declare2text(t)}")
  print(f"STLCG: {my_parser.transform_declare2STLCG(t)}")
  print()
```
(Output):<br />
Text: F\[0,2\](a\[t\]>1)<br />
STLCG: ♢ \[0, 2\]( a >= 1.0 )<br />

Text: G\[0,m\](a\[t\]>2 -> F\[5,8\](dis(d\[t\])))<br />
STLCG: (◻ \[0, inf\]( a >= 2.0 )) => (♢ \[5, 8\]( d == 1.0 ))<br /><br /><br />

```
########## Integration with STLCG Example 1 ##########
my_data = [0.2, 0.99, 0.6, 0.3, 0.99, 0.6]
my_array = np.array(my_data)
a_np = my_array.reshape([1, my_array.shape[0],1])
a = torch.tensor(a_np, requires_grad=False) 
input = (a.flip(1))
pscale = 1
scale = 0

#start of actual example
formula_str="Existence[0,m](a[t]>1)"
formula = my_parser.transform_declare2STLCG(formula_str)
print(formula_str)
print(formula)
print(formula.robustness_trace(input, pscale=pscale, scale=scale).flip(1))
print()
```
(Output):<br/>
Existence\[0,m\](a\[t\]>1)<br />
♢ \[0, inf\]( a >= 1.0 )<br />
tensor(\[\[[-0.0100\],<br />
         \[-0.0100\],<br />
         \[-0.0100\],<br />
         \[-0.0100\],<br />
         \[-0.0100\],<br />
         \[-0.4000\]\]\], dtype=torch.float64)<br /><br /><br />

```
########## Integration with STLCG Example 2 ##########
formula_str="Response[0,m][5,8](a[t]>1,dis(d[t]))" #note that "dis(d[t])" is checking if d==1
formula = my_parser.transform_declare2STLCG(formula_str)
print(formula_str)
print(formula)
print(formula.robustness_trace((input,input), pscale=pscale, scale=scale).flip(1)) #for stlcg note that you have to provide a tuple as input here
```
(Output):<br/>

Response\[0,m\]\[5,8\](a\[t\]>1,dis(d\[t\]))<br />
(◻ \[0, inf\]( a >= 1.0 )) => (♢ \[5, 8\]( d == 1.0 ))<br />
tensor(\[\[\[0.8000\],<br />
         \[0.7000\],<br />
         \[0.7000\],<br />
         \[0.7000\],<br />
         \[0.4000\],<br />
         \[0.4000\]\]\], dtype=torch.float64)
