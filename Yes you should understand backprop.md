# Yes you should understand backprop
(_Dịch từ bài đăng của Andrej Karpathy trên Medium_)
[Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)


Khi chúng tôi dạy khóa học CS231n tại Stanford, chúng tôi cố ý thiết kế các bài tập lập trình bao gồm các tính toán cụ thể liên quan backpropagation ở mức thấp nhất. Sinh viên phải triển khai đến cả forward và backward pass tại mỗi layer bằng thuần numpy. Một điều không thể tránh khỏi, một vài sinh viên rằng:
> "Tại sao chúng ta lại phải viết code cho phần backward trong khi các frameworks hiện nay, như TensorFlow, đã tính chúng một cách tự động?"

Đây dường như là lời phàn nàn hoàn toàn hợp lý - nếu như bạn không bao giờ viết phần backward sau khi khóa học kết thúc thì tại sao phải tập viết chúng? Liệu có phải chúng tôi chỉ đang tra tấn sinh viên cho vui? Một số câu trả lời có thể dễ dàng được đồng thuận kiểu như "Đáng để biết thứ gì thực sự xáy ra bên dưới" hoặc là: "Có thể sau này bạn sẽ muốn cải tiến trên thuật toán trung tâm ", nhưng có 1 lý do mạnh mẽ và thực tế hơn mà tôi muốn đề cập đến ở post này:

##### **Vấn đề với Backpropagation là trừu tượng hóa không cẩn thận (leaky abstraction)** 

Nói cách khác, rất dễ rơi vào cái bẫy của sự trừu tượng hóa quý trình học-tin tưởng rằng chỉ cần stack một cách tùy ý các layer với nhau và backprop sẽ bằng cách thần kỳ nào đó làm cho chúng hoạt động trên dữ liệu của bạn.

### **Vanishing gradients on sigmoids(Sự triệt tiêu của đạo hàm hay đạo hàm = 0)**

Tại một thời điểm, sử dụng phi tuyến sigmoid (hoặc tanh) trong fully connected layers đã trở nên hợp thời(fashionable??). Phần khó khăn mà mọi người có thể không nhận ra cho đến khi nghĩ đến backward pass là nếu bạn khởi tạo bộ trọng số hoặc tiền xử lý dữ liệu dữ liệu một cách cẩu thả thì các hàm phi tuyến của bạn sẽ trở nên "bão hòa" và dừng học - loss sẽ đi ngang và không giảm nữa. Ví dụ, một fully connected layer với hàm phi tuyến sigmoid:
```
z = 1/(1 + np.exp(-np.dot(W, x))) # forward pass
dx = np.dot(W.T, z*(1-z)) # backward pass: local gradient for x
dW = np.outer(z*(1-z), x) # backward pass: local gradient for W
```
Nếu ma trận trọng số **W** được khởi tạo với giá trị quá lớn, output của phép nhân ma trận sẽ có khoảng giá trị rất lớn (ví dụ trong khoảng -400 đến 400), điều này làm cho giá trị của **z** gần như là giá trị nhị phân (1 hoặc 0). Khi đó đạo hàm của hàm sigmoid là **z\*(1- z)** sẽ = **0** ("vannish"), khiến cho đạo hàm với cả **x** và **W** cũng = **0**. Phần còn lại của quá trình backward từ điểm này cũng = **0** do chain rule.
![Image](https://miro.medium.com/max/3000/1*gkXI7LYwyGPLU5dn6Jb6Bg.png)

Một fun fact khác về hàm sigmoid đó là đạo hàm cảu nó đạt cực đại = **0.25** khi **z = 0.5** . Điều đó có nghĩa là mỗi khi đi qua 1 sigmoid gate thì độ lớn của đạo hàm toàn bộ sẽ giảm còn 1/4 hoặc ít hơn. Nếu sử dụng SGD cơ bản, điều này sẽ khiến cho những layer thấp hơn của mạng học chậm hơn so với layer cao hơn.

**TLDR**: Nếu bạn đang sử dụng sigmoid hoặc tanh trong mạng và bạn hiểu rõ về backprop, bạn luôn luôn nên chắc chắn rằng việc khởi tạo không gây ra sự bão hòa.
