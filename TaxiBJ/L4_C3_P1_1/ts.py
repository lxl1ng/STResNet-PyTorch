import matplotlib
from TaxiBJ.experimentTaxiBJ import *
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
nb_epoch = 20  # number of epoch at training stage
batch_size = 32  # batch size
T = 48  # number of time intervals in one day
len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of  peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
days_test = 7 * 4
len_test = T * days_test
map_height, map_width = 32, 32  # grid size
nb_flow = 2
lr = 0.0002  # learning rate
nb_residual_unit = 4

X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
    load_data(len_closeness=3, len_period=1, len_trend=1, len_test=28 * 48)
# print(module)
X_test_torch = [torch.Tensor(x) for x in X_test]
torch.Tensor(Y_test)
# print(X_test_torch)
# 导入模型
print("model loading")

path_model = "best.pt"
model = STResNet(
        learning_rate=lr,
        epoches=nb_epoch,
        batch_size=batch_size,
        len_closeness=len_closeness,
        len_trend=len_trend,
        external_dim=external_dim,
        map_heigh=map_height,
        map_width=map_width,
        nb_flow=nb_flow,
        nb_residual_unit=nb_residual_unit,
        data_min=mmn._min,
        data_max=mmn._max)
model.load_state_dict(torch.load(path_model), False)
# out=model.evaluate(X_test_torch, Y_test_torch)
# model.evaluate(X_test_torch, Y_test_torch)
print("finish")
# print(out)
# path_model = "best.pt"
print('!!!!!!!!!!!!!!!!!!!!')
# print(X_test_torch[0].size())
# out=model.f
model.eval()
print(model)
with torch.no_grad():
    output = model.forward(X_test_torch[0], X_test_torch[1], X_test_torch[2], X_test_torch[3])
# print(output)
# print(output.size())
selected_plane = 0
i = 0
for ten in output:
    selected_plane += output[i, 0, :, :]+output[i, 1, :, :]
    print(i)
    i += 1
# selected_plane = output[1, 0, :, :] + output[1, 0, :, :]
print(selected_plane)
# selected_plane= torch.Tensor(Y_test)[0, 0, :, :]
# print(selected_plane)
# 使用 imshow 显示所选平面
plt.imshow(selected_plane, cmap='RdYlGn_r', interpolation='nearest')

# 添加标题和其他可视化选项
plt.title('Traffic Flow for the First Channel')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 显示图像
plt.show()
