---
sticker: ""
tags:
  - workshop
  - advancedAI
---

## Task 1
tameroon


| Mark         | 1   | 2:1 | 2:2 | 3   | Fail |
| ------------ | --- | --- | --- | --- | ---- |
| Num Students | 4   | 10  | 12  | 5   | 3    | 

```Math
TotalStudents = 4 + 10 + 12 + 5 + 3
```

Probability of 1st?

```math
First = 4/34
```

```math
UppSec = 10/34
```



|        | pass | fail |
| ------ | ---- | ---- |
| male   | 20   | 2    |
| female | 11   | 1    |

What is the probability of passing the module from this table?

```math
Pass = (20 + 11) / 34
```


What is the probability of being Female and passing? 

```math
Pass = (11) / 34
```


#### [Marginal Probabilities](Quantifying%20Uncertainty.md#Marginal%20Probabilities)
|      | sunny | rainy |
| ---- | ----- | ----- |
| hot  | 0.3   | 0.1   |
| cold | 0.1   | 0.5   | 

##### h. **Calculate the marginal probability of P(sunny)**
$$P(sunny)=P(sunny, hot)+P(sunny,cold)$$
```math
PSunny = 0.3 + 0.1
```

##### j. Calculate the marginal probability P(hot)

```math
pHot = 0.3+0.1
```

##### k. Calculate the marginal probability P(hot|sunny)
this is a case of [conditional probabilities](Quantifying%20Uncertainty.md#Conditional%20Probabilites) 

$$P(A|B) = P(A∧B) = P(B | A) * \frac{P(A)}{P(B)}*$$

$$ P(hot | sunny) = \frac{P(hot ∧ sunny)}{P(sunny)}$$
$$ P(hot|sunny) = \frac{0.3}{0.4}$$
```math
pHotSunny = 0.3/0.4 
```

##### k. Calculate the marginal probability P(rainy|cold)

$$P(rainy|cold) = \frac{P(rainy ∧ cold)}{P(cold)}$$

```math
pRainyCold = 0.5/0.6
```


| X   | Y   | P(X,Y) |
| --- | --- | ------ |
| x   | y   | 0.2    |
| x   | ¬y  | 0.3    |
| ¬x  | y   | 0.4    |
| ¬x  | ¬y  | 0.1    | 

##### l. P(x ∧ y)
0.2

##### m. P(x)
0.5
##### n. P(x ∨ y)

0.2 + 0.3  + 0.4
##### o. P(y)
0.6
##### p. P(x|y)
$$ \frac{P(x ∧ y)}{P(y)}$$
```math
(0.2)/0.6
```


##### q. P(¬x|y)

$$ \frac{P(¬X ∧ Y)}{P(Y)}$$

```math
0.4/0.6
```
##### r. P(¬y|x)

```math
0.3/0.5
```



##### Given the following Probability distributions
| S      | T    | W    | Probability |
| ------ | ---- | ---- | ----------- |
| Summer | hot  | sun  | 0.30        |
| Summer | hot  | rain | 0.05        |
| Summer | cold  | sun  | 0.10        |
| Summer | cold | rain | 0.05        |
| Winter | hot | sun  | 0.10        |
| Winter | hot  | rain | 0.05        |
| Winter | cold  | sun  | 0.15        |
| Winter| cold | rain | 0.20        |

##### s. P(sun)
```math
0.3+0.1+0.1+0.15
```
##### t. P(sun|winter)
$$P(Sun ∧ Winter) = 0.10 + 0.15$$
$$P(Winter) = 0.10 + 0.05  0.15 + 0.20 = 0.50$$

```math
(0.10 + 0.15) / 0.50
```
##### u. P(sun|winter, hot)
given it is winter and hot, what is the probability it is sunny  =>  P(sun | (winter && hot))

$$ P(Winter ∧ hot) = 0.10 + 0.05 = 0.15$$

$$ P(s|W∧h) = \frac{P(s ∧ (W∧h))}{P(w∧h)}$$
$$ P(s|W∧h) = \frac{P(s ∧ W∧h)}{P(W∧h)} = \frac{0.10}{0.15}$$

```math
0.10 / 0.15
```




##### Given the following Probability distributions
| Rash | Measles | P(X, Y)      |
| ---- | ------- | ------------ |
| r    | m       | P(r,m) = 0.1 |
| r    | ¬m      | P(r,m)=0.8   |
| ¬r   | m       | P(¬r,m)=0.01 |
| ¬r   | ¬m      | P(¬r,m)=0.09 | 



What is the probability of not having measles given that a person has a rash? 
P(not having measles | person has a rash)
$$P(¬m | r) = \frac{P(¬m ∧ r )}{P(r)} = \frac{0.8}{0.8 + 0.1}$$
```math
(0.8) / (0.8 + 0.1)
```



What is the probability of having measles given that a person has a rash? 

$$P(m|r) = \frac{0.1}{0.9}$$

```math
( 0.1 ) / ( 0.9 )
```

## Task 2 

##### more info on bayes stuff
3blue1brown on baysean #watch
true positive true negative etc etc






##### Task 2a
Consider the following fictitious scientific information. 

Doctors find that people with the Kreuzfeld-Jacob disease (KJ) almost invariably ate hamburgers, thus P(HamburgerEater|KJ) = 0.9.
The probability of an individual having KJ is rather low, about 1/100, 000. 


Assuming eating lots of hamburgers is rather widespread, say P(HamburgerEater) = 0.5, what is the probability that a Hamburger Eater will have the KJ disease?

$$P(HamburgerEater|KJ) = 0.9$$
$$ P(KJ) = 1/100000 $$
**If**
$$P(HamburgerEater) = 0.5$$
**what is Probability of having KJ given they are a Hamburger Eater?**
$$P(KJ|HambergerEater)$$


| HamburgerEater | KJ  |     |
| -------------- | --- | --- |
| HE             | KJ  |     |
| HE             | ¬KJ |     |
| ¬HE            | KJ  |     |
| ¬HE | ¬KJ |     |

$$P(KJ|HE) = \frac{P(KJ ∧ HE)}{P(HE)}$$


```math
(0.9 *(1/1000))/ (0.5)
```


```math
probTestAndDis = 0.99 * (1/1000)
probNotTestAndNotDis = 0.95 * (1 - 1/1000)
probDisAndNotTest = (1/1000) - 0.001

```


![[Pasted image 20230928190836.png]]


#### answer for 2b
0.001978






