# B830：Black-Scholes-Merton期权定价模型原理与Python实现

> **作者：** 史云慧 (XXXX大学)    
> **邮箱：** <shiyunhui@example.com> 

> **Source**：本文参考了如下资料，特此致谢！  
> Islam, R. (2023). Pricing Derivatives Using Black-Scholes-Merton Model. *Statistics and Probability*. [Link](https://mrislambd.github.io/statandprob/posts/optionprice/), [GitHub](https://github.com/mrislambd).

&emsp; 

- **Title**: Black-Scholes-Merton期权定价模型原理与Python实现
- **Keywords**: Black-Scholes, 期权定价, Python, 金融工程, 随机微积分, Greeks

&emsp; 

>**声明**：本文写作过程中借助了 AI 工具，但内容经过了严格的审核和润色，确保准确传达了原论文的核心思想和方法。[Claude 对话链接](https://claude.ai/share/example-conversation-id)

---

**Note-助教须知**：&#x1F34E; 

- 下文内容基于原文翻译和适当补充，确保技术准确性和中文表达的自然性
- Python代码已根据连享会推文风格进行了优化和完善
- 数学公式推导经过仔细核对，确保准确性
- 所有技术术语均采用业界标准中文翻译

# 中文精要：Islam-2023-BSM

## 1. 简介

在金融衍生品定价领域，Black-Scholes-Merton（BSM）模型无疑是最具影响力的理论突破之一。该模型为期权等衍生品的定价提供了封闭解，成为量化金融的基石。本文将深入探讨BSM模型的理论基础、数学推导过程，并提供完整的Python实现。

**面向读者**：

本文特别适合以下读者：

* 金融工程、量化金融方向的学生和从业者
* 具备基础微积分和概率论知识的研究人员  
* 对期权定价感兴趣的Python程序员
* 希望深入理解BSM模型数学原理的学习者

**学完本推文，你将掌握**：

* Black-Scholes-Merton模型的理论基础和数学推导过程
* 几何布朗运动和Ito微积分的核心概念和应用
* 完整的Python实现和工业级代码设计模式
* 希腊字母（Greeks）的计算方法和实际意义
* 模型的实际应用场景和局限性分析

**关键 Python 操作**：

- 数学计算：`numpy`, `scipy.stats`
- 面向对象设计：`@dataclass`
- 核心函数：`bsm()`, `delta()`, `gamma()`
- 验证工具：`check_put_call_parity()`

## 2. 理论基础

在现代金融工程中，Black-Scholes-Merton模型的成功建立离不开对随机过程的深入理解。本节将从布朗运动出发，逐步推导至BSM偏微分方程。

### 2.1 布朗运动与几何布朗运动

**布朗运动**（Brownian Motion）是随机过程理论中的核心概念，由植物学家Robert Brown发现并命名。在金融建模中，我们通常假设资产价格服从**几何布朗运动**（Geometric Brownian Motion, GBM）。

GBM的随机微分方程形式为：

$$
dS_t = \mu S_t dt + \sigma S_t dB_t
$$

其中：
- $S_t$：时刻 $t$ 的资产价格
- $\mu$：期望收益率（漂移项）
- $\sigma$：波动率（扩散项）
- $B_t$：标准布朗运动

为求解这个随机微分方程，我们应用Ito公式到函数 $Z_t = \ln(S_t)$。通过Taylor展开和Ito微积分规则，得到：

$$
d\ln(S_t) = \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma dB_t
$$

积分后得到解：

$$
\ln\left(\frac{S_t}{S_0}\right) = \left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma B_t
$$

最终得到资产价格的显式表达：

$$
S_t = S_0 \exp\left\{\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma B_t\right\}
$$

### 2.2 Black-Scholes-Merton偏微分方程

BSM模型的核心思想是通过构建无风险投资组合来消除随机性。考虑一个期权 $V(S,t)$，其价格依赖于标的资产价格 $S$ 和时间 $t$。

根据Ito引理，期权价格的微分形式为：

$$
dV = \left(\mu S \frac{\partial V}{\partial S} + \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right)dt + \sigma S \frac{\partial V}{\partial S} dB_t
$$

现在构建一个无风险投资组合：做空一个期权，做多 $\frac{\partial V}{\partial S}$ 份标的资产。组合价值为：

$$
\Pi = -V + \frac{\partial V}{\partial S}S
$$

组合的价值变化为：

$$
d\Pi = -dV + \frac{\partial V}{\partial S}dS
$$

代入上述表达式并消去随机项 $dB_t$，得到：

$$
d\Pi = \left(-\frac{\partial V}{\partial t} - \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right)dt
$$

在无套利假设下，该组合的收益率应等于无风险利率 $r$：

$$
d\Pi = r\Pi dt = r\left(-V + \frac{\partial V}{\partial S}S\right)dt
$$

联立以上两式，最终得到著名的BSM偏微分方程：

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

## 3. 数学求解

BSM偏微分方程的求解是一个经典的数学物理问题。通过巧妙的变量替换，我们可以将其转化为热传导方程，从而得到解析解。

### 3.1 热传导方程变换

为了简化BSM偏微分方程，我们进行以下变量替换：

- $S = Ke^x$，其中 $x = \ln(S/K)$
- $t = T - \frac{2\tau}{\sigma^2}$，所以 $\tau = \frac{1}{2}\sigma^2(T-t)$
- $V(S,t) = Kv(x,\tau)$

对原方程进行变量替换，计算各阶导数：

**一阶导数变换**：
$$
\frac{\partial V}{\partial S} = \frac{\partial V}{\partial x}\frac{\partial x}{\partial S} = \frac{K}{S}\frac{\partial v}{\partial x}
$$

**二阶导数变换**：
$$
\frac{\partial^2 V}{\partial S^2} = \frac{K}{S^2}\left(\frac{\partial^2 v}{\partial x^2} - \frac{\partial v}{\partial x}\right)
$$

**时间导数变换**：
$$
\frac{\partial V}{\partial t} = -\frac{\sigma^2}{2}\frac{\partial V}{\partial \tau}
$$

将这些变换代入原BSM方程，经过化简得到：

$$
\frac{\partial v}{\partial \tau} = \frac{\partial^2 v}{\partial x^2} + \frac{2r}{\sigma^2}\frac{\partial v}{\partial x} - \frac{2r}{\sigma^2}v
$$

为进一步简化，令 $v(x,\tau) = e^{\alpha x + \beta\tau}u(x,\tau)$，选择适当的 $\alpha$ 和 $\beta$ 可消去一阶导数项和零阶项，最终得到标准热传导方程：

$$
\frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2}
$$

### 3.2 最终定价公式

热传导方程的解可以通过Fourier方法得到。经过逆向变换，我们最终得到欧式期权的定价公式：

**欧式看涨期权**：

$$
C(S,t) = Se^{-q(T-t)}N(d_1) - Ke^{-r(T-t)}N(d_2)
$$

**欧式看跌期权**：

$$
P(S,t) = Ke^{-r(T-t)}N(-d_2) - Se^{-q(T-t)}N(-d_1)
$$

其中：

$$
d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T-t}
$$

这里：
- $N(\cdot)$：标准正态分布的累积分布函数
- $S$：标的资产当前价格
- $K$：期权行权价
- $T-t$：到期时间（年）
- $r$：无风险利率（连续复利）
- $q$：连续分红率
- $\sigma$：波动率

## 4. Python实现

本节将提供完整的Python实现，包括面向对象设计、核心定价函数和风险指标计算。

### 4.1 环境准备与类设计

首先导入必要的库并定义数据结构：

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass
class Equity:
    """标的资产类
    
    Attributes:
        spot: 现货价格
        dividend_yield: 连续分红率
        volatility: 波动率
    """
    spot: float
    dividend_yield: float
    volatility: float

@dataclass  
class EquityOption:
    """期权类
    
    Attributes:
        strike: 行权价
        time_to_maturity: 到期时间（年）
        put_call: 期权类型（'call' 或 'put'）
    """
    strike: float
    time_to_maturity: float
    put_call: str

@dataclass
class EquityForward:
    """远期合约类"""
    strike: float
    time_to_maturity: float
```

### 4.2 BSM定价核心函数

```python
def bsm(underlying: Equity, option: EquityOption, rate: float) -> float:
    """
    Black-Scholes-Merton期权定价函数
    
    Parameters:
    ---------
    underlying : Equity
        标的资产对象
    option : EquityOption  
        期权对象
    rate : float
        无风险利率
        
    Returns:
    --------
    float : 期权价格
    """
    S = underlying.spot
    K = option.strike
    T = option.time_to_maturity
    r = rate
    q = underlying.dividend_yield
    sigma = underlying.volatility
    
    # 处理极限情况：当行权价接近0时
    if K < 1e-8:
        if option.put_call.lower() == "call":
            return S  # 看涨期权价值等于现货价格
        else:
            return 0.0  # 看跌期权价值为0
    
    # 计算d1和d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 根据期权类型计算价格
    if option.put_call.lower() == "call":
        price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                K * np.exp(-r * T) * norm.cdf(d2))
    elif option.put_call.lower() == "put":
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("期权类型必须是 'call' 或 'put'")
    
    return price
```

### 4.3 希腊字母计算

```python
def delta(underlying: Equity, option: EquityOption, rate: float) -> float:
    """
    计算Delta：期权价格对标的资产价格的敏感度
    使用中心差分法计算
    """
    bump = 0.01 * underlying.spot  # 1%的扰动
    
    # 构建扰动后的标的资产对象
    bumped_up = Equity(
        spot=underlying.spot + bump,
        dividend_yield=underlying.dividend_yield,
        volatility=underlying.volatility
    )
    
    bumped_down = Equity(
        spot=underlying.spot - bump,
        dividend_yield=underlying.dividend_yield,
        volatility=underlying.volatility
    )
    
    # 计算中心差分
    price_up = bsm(bumped_up, option, rate)
    price_down = bsm(bumped_down, option, rate)
    
    return (price_up - price_down) / (2 * bump)

def gamma(underlying: Equity, option: EquityOption, rate: float) -> float:
    """
    计算Gamma：Delta对标的资产价格的敏感度
    即期权价格的二阶导数
    """
    bump = 0.01 * underlying.spot
    
    bumped_up = Equity(
        spot=underlying.spot + bump,
        dividend_yield=underlying.dividend_yield,
        volatility=underlying.volatility
    )
    
    bumped_down = Equity(
        spot=underlying.spot - bump,
        dividend_yield=underlying.dividend_yield,
        volatility=underlying.volatility
    )
    
    original_price = bsm(underlying, option, rate)
    price_up = bsm(bumped_up, option, rate)
    price_down = bsm(bumped_down, option, rate)
    
    # 二阶中心差分
    return (price_up - 2 * original_price + price_down) / (bump**2)

def fwd(underlying: Equity, forward: EquityForward, rate: float) -> float:
    """计算远期价格"""
    S = underlying.spot
    K = forward.strike
    T = forward.time_to_maturity
    r = rate
    q = underlying.dividend_yield
    
    forward_price = S * np.exp((r - q) * T) - K
    return forward_price

def check_put_call_parity(
    underlying: Equity, 
    call_option: EquityOption, 
    put_option: EquityOption, 
    rate: float
) -> bool:
    """
    验证看跌-看涨平价关系
    C - P = S*exp(-qT) - K*exp(-rT)
    """
    call_price = bsm(underlying, call_option, rate)
    put_price = bsm(underlying, put_option, rate)
    
    S = underlying.spot
    K = call_option.strike
    T = call_option.time_to_maturity
    r = rate
    q = underlying.dividend_yield
    
    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    
    return np.isclose(parity_lhs, parity_rhs, atol=1e-4)
```

## 5. 实际应用示例

下面我们通过具体的数值例子来演示BSM模型的应用。

### 5.1 期权定价计算

```python
if __name__ == "__main__":
    # 示例1：计算看涨期权价格
    # 标的资产：现货价格450，分红率1.4%，波动率14%
    eq = Equity(450, 0.014, 0.14)
    # 期权参数：行权价470，期限0.23年，看涨期权
    option_call = EquityOption(470, 0.23, "call")
    
    # 无风险利率5%
    call_price = bsm(eq, option_call, 0.05)
    print(f"看涨期权价格: {call_price:.4f}")  # 输出: 5.8340
    
    # 示例2：极限情况分析 - 当行权价趋近于0时
    # 理论上，当K→0时，看涨期权价值应趋近于现货价格
    call_price_limit = bsm(eq, EquityOption(1e-15, 0.26, "call"), 0.0)
    print(f"行权价趋近0时的看涨期权价格: {call_price_limit:.4f}")  # 输出: 450.0
    
    # 示例3：计算看跌期权价格
    option_put = EquityOption(500, 0.26, "put")
    # 特殊情况：无分红，极低波动率
    put_price = bsm(Equity(450, 0.0, 1e-9), option_put, 0.0)
    print(f"看跌期权价格: {put_price:.4f}")  # 输出: 50.0
    
    # 示例4：验证看跌-看涨平价关系
    eq_parity = Equity(450, 0.015, 0.15)
    call_opt = EquityOption(470, 0.26, "call")
    put_opt = EquityOption(470, 0.26, "put")
    
    parity_check = check_put_call_parity(eq_parity, call_opt, put_opt, 0.05)
    print(f"看跌-看涨平价关系验证: {parity_check}")  # 输出: True
```

### 5.2 希腊字母分析

```python
# 计算希腊字母
call_delta = delta(eq, option_call, 0.05)
call_gamma = gamma(eq, option_call, 0.05)

print(f"Delta: {call_delta:.4f}")  # Delta值
print(f"Gamma: {call_gamma:.6f}")  # Gamma值
```

## 6. 模型分析与讨论

### 6.1 极限行为分析

分析BSM公式的极限行为有助于深入理解模型的金融含义：

**当行权价 $K \to 0$ 时**：
- $\ln(S_0/K) \to \infty$，导致 $d_1 \to \infty$ 且 $d_2 \to \infty$
- 累积分布函数 $N(d_1) \to 1$ 且 $N(d_2) \to 1$
- 看涨期权价格 $C \to S_0e^{-qT}$，看跌期权价格 $P \to 0$

这符合金融直觉：当行权价接近0时，看涨期权几乎肯定会被执行，其价值接近标的资产的现值。

**当行权价 $K \to \infty$ 时**：
- $d_1 \to -\infty$，$d_2 \to -\infty$
- $N(d_1) \to 0$，$N(d_2) \to 0$
- 看涨期权价格 $C \to 0$，看跌期权价格 $P \to Ke^{-rT}$

### 6.2 模型局限性

尽管BSM模型具有理论上的优美性，但在实际应用中需要注意其局限性：

1. **常数波动率假设**：模型假设波动率是常数，但实际市场中波动率是时变的，存在波动率微笑现象
2. **连续交易假设**：理论上需要连续调整对冲组合，但现实中只能离散交易
3. **对数正态分布假设**：假设资产收益率服从对数正态分布，但实际收益率分布可能具有尖峰厚尾特征
4. **无摩擦市场假设**：忽略了交易成本、税收、卖空限制等现实因素
5. **欧式期权限制**：模型仅适用于欧式期权，对美式期权需要其他方法

### 6.3 进一步学习资源

**经典教材**：
- Hull, J.C. (2022). *Options, Futures, and Other Derivatives*. Pearson.
- Björk, T. (2004). *Arbitrage Theory in Continuous Time*. Oxford.

**学术专著**：
- Shreve, S.E. (2004). *Stochastic Calculus for Finance II*. Springer.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*. Wiley.

**在线资源**：
- [QuantStart](https://www.quantstart.com/)：量化金融教程
- [Quantopian Lectures](https://github.com/quantopian-lectures/lectures)：量化交易讲座

## 7. 总结

Black-Scholes-Merton模型为期权定价提供了理论基础和实用工具。通过本文的学习，我们不仅掌握了模型的数学推导过程，还实现了工业级的Python代码。

**核心要点回顾**：

* **理论基础**：BSM模型基于几何布朗运动假设，通过Ito微积分和无套利原理推导
* **数学工具**：运用偏微分方程和热传导方程的求解技巧
* **编程实现**：采用面向对象设计，提供完整的定价和风险管理功能
* **实际应用**：包含希腊字母计算、平价关系验证等实用功能

虽然BSM模型存在一些假设限制，但它仍然是金融工程领域最重要的理论成果之一，为后续更复杂的定价模型（如局部波动率模型、随机波动率模型）奠定了基础。

建议读者在实际应用中结合市场数据验证模型效果，并根据具体需求选择合适的改进模型。