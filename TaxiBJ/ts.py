import matplotlib

matplotlib.use('TkAgg')
from data.TaxiBJ.TaxiBJ import *
from models.STResNet import *

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # 定义你的神经网络结构
#
#     def forward(self, x):
#         # 定义前向传播过程
#         return x


X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
    load_data(len_closeness=3, len_period=1, len_trend=1, len_test=28 * 48)
# print(module)
X_test_torch = [torch.Tensor(x) for x in X_test]
torch.Tensor(Y_test)
# print(X_test_torch)
# 导入模型
print("model loading")

path_model = "best.pt"
model = STResNet()
model.load_model("best")
# out=model.evaluate(X_test_torch, Y_test_torch)
# model.evaluate(X_test_torch, Y_test_torch)
print("finish")
# print(out)
# path_model = "best.pt"
print('!!!!!!!!!!!!!!!!!!!!')
# print(X_test_torch[0].size())
# out=model.f
model.eval()
with torch.no_grad():
    output = model.forward(X_test_torch[0], X_test_torch[1], X_test_torch[2], X_test_torch[3])
# print(output)
# print(output.size())
# selected_plane = 0
i = 0
# for ten in output:
#     selected_plane += output[i, 0, :, :]
#     selected_plane += output[i, 1, :, :]
#     print(i)
#     i += 1
selected_plane = output[0, 0, :, :] + output[0, 1, :, :]
# selected_plane= torch.Tensor(Y_test)[0, 0, :, :]
# print(selected_plane)
# 使用 imshow 显示所选平面
plt.imshow(selected_plane, cmap='viridis',interpolation='nearest' )

# 添加标题和其他可视化选项
plt.title('Traffic Flow for the First Channel')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 显示图像
plt.show()
