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

### Dying ReLUs

Một hàm phi tuyến thú vị khác là ReLU, threshold các neuron tại 0. Forward và backward cho 1 fully connected layer sử dụng ReLU sẽ gồm phần chính như sau:
```
z = np.maximum(0, np.dot(W, x)) # forward pass
dW = np.outer(z > 0, x) # backward pass: local gradient for W
```
Nếu bạn nhìn vào 2 dòng code này, bạn sẽ nhận ra rằng, nếu 1 neuron không kích hoạt ở forward pass (z = 0) thì đạo hàm của W sẽ = 0. Điều này có thể dẫn đến vấn đề gọi là "Dead ReLU", trong trường hợp đó, một neuron ReLU không may được khởi tạo theo cách khiến cho nó không bao giờ kích hoạt ("fire"), hoặc trọng số của 1 neuron bị knocked off trong quá trình huấn luyện, thì neuron này sẽ chết hẳn. Nó giống như một tổn thương não vĩnh viễn, không thể hồi phục. Đôi khi bạn có thể forward toàn bộ dữ liệu huấn luyện qua 1 mạng đã huấn luyện và phát hiện ra rằng tỉ lệ lớn neuron = 0 trong suốt quá trình.
![Image](https://miro.medium.com/max/875/1*g0yxlK8kEBw8uA1f82XQdA.png)

**TLDR**: Nếu bạn hiểu được backprop và sử dụng ReLu trong mạng của mình, bạn luôn luôn nên để ý đến dead ReLUs. Có những neuron không bao giờ kích hoạt với bất kỳ mẫu nào trong tập huấn luyện, và sẽ chết hẳn. Các neuron cũng có thể chết trong quá trình huấn luyện, thường thì nó là dấu hiệu của việc learning rates quá lớn.

### Exploding gradients in RNNs
Vanilla RNNs là một ví dụ tốt khác cho những ảnh hưởng không trực quan của backprop. Tôi sẽ lấy 1 slide từ CS231n có 1 phiên bản đơn giản của RNN, không nhận bất kỳ input x nào và chỉ tính recurrence trên hidden state (tương đương với việc đầu vào x có thể luôn = 0):
![Image](https://miro.medium.com/max/3000/1*dqlX0ixpk1O3225bZ1LGnA.png)

Mạng RNN này được unroll cho T time steps (???). Khi nhìn vào phần backward, bạn sẽ thấy tín hiệu đạo hàm đi qua tất cẩ hidden state thì luôn được nhân với cùng 1 ma trận(ma trận recurrence **Whh**), xen kẽ với phi tuyến backprop.

Điều gì xảy ra khi bạn lấy 1 số **a** và bắt đầu nhân nó với một số số **b* khác(a * b * b...)?  Chuỗi này sẽ tiến về 0 khi **|b| < 1** và tiến ra vô cùng khi **|b| > 1** . Điều tương tự cũng đang xảy ra với backward pass của RNN, ngoại trừ việc **b** là 1 ma trận cho nên thay vào đó chúng ta phải suy nghĩ về giá trị riêng lớn nhất của nó.

**TLDR**: Nếu bạn hiểu backprop và đang sử dụng RNN, bạn nên suy nghĩ về việc gradient clipping, hoặc sử dụng LSTM

### Spotted in the Wild: DQN Clipping
Thêm 1 ví dụ nữa - thứ thực sự truyền cảm hứng cho post này. Hôm qua, tôi search web cách triển khai Deep Q Learning của TensorFlow(để biết cách những người khác giải quyết việc tính toán numpy equivalent của **Q[:, a]**, trong đó **a** là 1 vector số nguyên - hóa ra phép toán tầm thường này không được hỗ trợ trong TF). Dù sao thì, tôi search "dqn tensorflow", click vào link đầu tiên và tìm core code, đây là 1 đoạn trích:
![Image](https://miro.medium.com/max/3000/1*pyz5lHFDho07cFzIA6tWmQ.png)

Nếu bạn quen với **DQN**, bạn có thể thấy there is the **target_q_t**, which is just **[reward * \gamma \argmax_a Q(s’,a)]** , and then there is **q_acted**, which is **Q(s,a)** of the action that was taken. The authors here subtract the two into variable **delta**, which they then want to minimize on line 295 with the L2 loss with **tf.reduce_mean(tf.square())**. So far so good.(Không biết dịch kiểu gì :< ).

Vấn đề ở dòng 291. Các tác giả đang cố giải quyết các trường hợp ngoại lệ, nếu **delta** quá lớn, họ clip nó với **tf.clip_by_value**. Việc này có chủ đích tốt và có vẻ hợp lý từ góc độ forward pass, nhưng nếu nhìn từ góc độ backward pass thì nó gây ra lỗi lớn.

Hàm **clip_by_value** có đạo cục bộ bằng 0 ngoài khoảng **min_delta** đến **max_delta**, do đó bất cứ khi nào delta vượt ngoài khoảng max/min_delta, đạo hàm = 0 trong suốt backprop. Các tác giả đang cắt raw Q delta trong khi họ cố cắt đạo hàm. Trong trường hợp đó, thứ chính xác phải làm là sử dụng Huber loss in place của tf.square:
```
def clipped_error(x):
    return tf.select(tf.abs(x) < 1.0, 
                   0.5 * tf.square(x), 
                   tf.abs(x) - 0.5) # condition, true, false
```
Nó hơi thô lỗ trong TensorFlow vì tất cả những gì chúng ta muốn làm là cắt gradient nếu nó ở trên ngưỡng, nhưng vì chúng ta không thể can thiệp trực tiếp với các gradient nên chúng ta phải thực hiện theo cách xác định Huber loss. Trong Torch, điều này sẽ đơn giản hơn nhiều.

### In conclusion
Backprop là 1 leaky abstraction; nó là 1 kế hoạch phân chia sự ảnh hưởng với những hệ quả không tầm thường. Nếu như bạn cố bỏ qua cách nó thực sự hoạt động bởi vì "TensorFlow tự động làm mạng của tôi học", bạn sẽ không sẵn sàng vật lộn với những nguy hiểm mà nó mang lại, và bạn sẽ xây dụng và sửa lỗi cho mạng neuron kém hiệu quả hơn.

Tin tốt là backprop không khó để hiểu nếu được trình bày đúng cách. Tôi có cảm xúc tương đối mạnh mẽ về chủ đề này, bởi với tôi 95% tài liệu về backprop ngoài kia đang trình bày sai về nó và chỉ lấp đầy trang viết với những phép toán một cách máy móc. Thay vào đó, tôi gợi ý [CS231n lecture on backprop](https://www.youtube.com/watch?v=i94OvYb6noo), Và nếu ban có thời gian, hãy làm [CS231n Assignments](https://cs231n.github.io/), ở đó bạn sẽ triển khai backprop một cách thủ công và giúp bạn củng cố lại hiểu biết của mình về backprop.
