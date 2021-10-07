# Backpropagation

Created: October 6, 2021 3:06 PM

![Untitled](Backpropagation%20c3d0f01c549545e892f13a83604ab91c/Untitled.png)

- Một số quy ước
    
    $l \in \{1, 2, \ldots , L\}$, số layer đánh số từ sau input.
    
    $x$ là đầu vào của mạng
    
    $$\begin{align*}
    x &\in \mathbb R^{N \times d^{(0)}}\\
    a^{(0)} &= x
    \end{align*}$$
    
    $z_i^{(l)}$ là đầu vào của unit thứ i tại layer thứ $l$
    
    $$z^{(l)} \in \mathbb R^{N \times d^{(l)}}$$
    
    $a_i^{(l)}$ là đầu ra của unit thứ i tại layer thứ $l$ (sau khi đi qua hàm active $f$)
    
    $$a^{(l)} \in \mathbb R^{N \times d^{(l)}}$$
    
    $W^{(l)}$ là bộ trọng số chuyển từ layer thứ $l - 1$ sang layer thứ $l$ 
    
    $$W^{(l)} \in R^{d^{(l - 1)} \times d^{(l)}}$$
    
    $W^{(l)}_j$ là dòng thứ j của ma trận $W^{(l)}$
    
    $$\begin{align*}
    W^{(l)}_j &\in R^{1 \times d^{(l)}}
    \end{align*}$$
    
    $\ell$  là loss function, trong trường hợp này là cross entropy.
    
    $$\ell = -\sum_{j = 1}^N\log(\mu_j) y_j^T$$
    
    $y$ là đầu ra mong muốn tương ứng với $x$, trong trường hợp này $y_j$ có dạng **one-hot**
    
    $$\begin{align*}
    y &\in \mathbb R^{N \times d^{(L)}}\\
    y_j &\in \mathbb R^{1 \times d^{(L)}}
    \end{align*}$$
    
    $\mu$ là đầu ra của hàm Softmax cho layer cuối cùng.
    
    $$\mu = \mathcal S(a^{(L)}) \in \mathbb R^{N \times d^{(L)}}$$
    
- B1: Tính đạo hàm của $\ell$ với $\mu$
    
    $$\begin{align*}
    \frac{\partial \ell}{\partial \mu_{jk}} &= -\frac {y_{jk}}{\mu_{jk}}\\
    \frac{\partial \ell}{\partial \mu_{j}} &= - \frac {y_{j}}{\mu_{j}}  = \begin{bmatrix} \frac{\partial \ell}{\partial \mu_{j1}}&\frac{\partial \ell}{\partial \mu_{j2}} & \ldots & \frac{\partial \ell}{\partial \mu_{jd^{(L)}}}\end{bmatrix} \in \mathbb R^{1 \times d^{(L)}}\\
    \frac{\partial \ell}{\partial \mu} &= -\frac{y}{\mu} =  \begin{bmatrix} \frac{\partial \ell}{\partial \mu_{1}}&\frac{\partial \ell}{\partial \mu_{2}} & \ldots & \frac{\partial \ell}{\partial \mu_{N}}\end{bmatrix}^T \in \mathbb R^{N \times d^{(L)}}
    \end{align*}$$
    
- B2: Tính đạo hàm của $\mu$ với $a^{(L)}$
    
    $\mu_{jk}$ là phần tử hàng j cột k của $\mu$. Tương tự với $a_{ji}$
    
    $$\begin{align*}
    \mu_{jk} = \mathcal S(a^{(L)}_{ji}) &= \frac {e^{a^{(L)}_{jk}}}{\sum_{i=1}^{d^{(L)}}e^{a_{ji}^{(L)}}}\\
    \frac{\partial\mu_{jk}}{\partial a^{(L)}_{ji}} &= \frac {\mathbb I(k = i)e^{a_{ji}^{(L)}}\times \sum_{i=1}^{d^{(L)}}e^{a_{ji}^{(L)}} - e^{a_{ji}^{(L)}} \times e^{a_{jk}^{(L)}}}{(\sum_{i=1}^{d^{(L)}}e^{a_{ji}^{(L)}}) ^ 2}\\
    &= \mu_{ji} (\mathbb I(i = k) - \mu_{jk})\\
    \frac{\partial\mu_j}{\partial a^{(L)}_j} &=\begin{bmatrix}(1-\mu_{j1})\mu_{j1} &-\mu_{j2}\mu_{j1} &\cdots&-\mu_{jL}\mu_{j1}\\ -\mu_{j1}\mu_{j2}&(1-\mu_{j2})\mu_{j2} &\cdots&-\mu_{jL}\mu_{j2}\\\vdots&\vdots&&\vdots\\-\mu_{j1}\mu_{jL} &-\mu_{j2}\mu_{jL} &\cdots&(1-\mu_{jL})\mu_{jL}\end{bmatrix}\\
    \frac{\partial\mu}{\partial a^{(L)}} &=  \begin{bmatrix}\frac{\partial\mu_1}{\partial a^{(L)}_1}& \frac{\partial\mu_2}{\partial a^{(L)}_2}&\ldots &\frac{\partial\mu_N}{\partial a^{(L)}_N}\end{bmatrix}^T \in \mathbb R^{N \times d^{(L)} \times d^{(L)}}
    \end{align*}$$
    
- B3: Tính đạo hàm của $\ell$  với $a^{(L)}$
    
    $$\begin{align*}
    \frac{\partial \ell}{\partial a^{(L)}} &= \frac{\partial \ell}{\partial \mu}\frac{\partial \mu}{\partial a^{(L)}}\\
    &= \begin{bmatrix}\frac{\partial \ell}{\partial \mu_{1}} \frac{\partial \mu_1}{\partial a_{1}^{(L)}} &\frac{\partial \ell}{\partial \mu_{2}} \frac{\partial \mu_2}{\partial a_{2}^{(L)}} & \ldots &\frac{\partial \ell}{\partial \mu_{N}} \frac{\partial \mu_N}{\partial a_{N}^{(L)}} \end{bmatrix}^T \in \mathbb R^{N \times d^{(L)}}
    \end{align*}$$
    
- B4: Tính đạo hàm của $a^{(l)}$ với $z^{(l)}$
    
    $$\begin{align*}
    \frac{\partial a^{(l)}}{\partial z^{(l)}} &= \begin{bmatrix}\frac{\partial a_1^{(l)}}{\partial z^{(l)}_1}& \frac{\partial a_2^{(l)}}{\partial z^{(l)}_2}&\ldots &\frac{\partial a_N^{(l)}}{\partial z^{(l)}_N}\end{bmatrix}^T \in \mathbb R^{N \times d^{(l)} \times d^{(l)}}\\
    \frac{\partial a_1^{(l)}}{\partial z^{(l)}_1}  &\in \mathbb R^{d^{(l)} \times d^{(l)}} \text{phụ thuộc vào hàm f là gì}
    \end{align*}$$
    
- B5: Tính đạo hàm của $z^{(l)}$ với $W^{(l)}$
    
    $$\begin{align*}
    \frac{\partial z^{(l)}_{jk}}{\partial W^{(l)}_{jk}} &= a^{(l-1)}_{jk}\\
    \frac{\partial z^{(l)}_{jk}}{\partial W^{(l)}_{:k}} &= a^{(l - 1)}_j \in \mathbb R^{1 \times d^{(l- 1)}}\\
    \frac{\partial z^{(l)}_{j}}{\partial W^{(l)}} &= a^{(l - 1)}_j \in \mathbb R^{1 \times d^{(l- 1)}}\\
    \frac{\partial z^{(l)}}{\partial W^{(l)}} &= a^{(l - 1)} \in \mathbb R^{N \times d^{(l- 1)}}
    \end{align*}$$
    
    Với $W_{:k}$ là cột thứ k của ma trận $W$
    
- B6: **Tính đạo hàm của $z^{(l)}$ với $a^{(l -1 )}$**
    
    $$\begin{align*}
    \frac{\partial z^{(l)}_{jk}}{\partial a^{(l - 1)}_{jk}} &= W^{(l)}_{jk}\\
    \frac{\partial z^{(l)}_{jk}}{\partial a^{(l - 1)}_{j}} &= W^{(l)}_{:k} \in \mathbb R^{d^{(l - 1)} \times 1}\\
    \frac{\partial z^{(l)}_{j}}{\partial a^{(l - 1)}_j} &= W^{(l)} \in \mathbb R^{d^{(l - 1)} \times d^{(l)}}\\
    \frac{\partial z^{(l)}}{\partial a^{(l - 1)}} &= W^{(l)} \in \mathbb R^{d^{(l - 1)} \times d^{(l)}}
    \end{align*}$$
    
- B7: Tính đạo hàm của $\ell$  với $z^{(l)}$, $a^{(l - 1)}$, $W^{(l)}$
    
    $$$$